import os
import pickle

import numpy as np

from preprocess.parse_csv import Mimic3Parser, Mimic4Parser
from preprocess.encode import encode_concept
from preprocess.build_dataset import split_patients
from preprocess.build_dataset import build_code_xy, build_heart_failure_y, build_real_data, build_real_next_xy, build_visit_x
from preprocess.auxiliary import generate_code_code_adjacent, real_data_stat, generate_code_levels, generate_procedure_levels
from config import get_preprocess_args


PARSERS = {
    'mimic3': Mimic3Parser,
    'mimic4': Mimic4Parser
}


if __name__ == '__main__':
    args = get_preprocess_args()

    data_path = args.data_path
    dataset = args.dataset

    dataset_path = os.path.join(data_path, dataset)
    raw_path = os.path.join(dataset_path, 'raw')
    if not os.path.exists(raw_path):
        os.makedirs(raw_path)
        print('please put the CSV files in `data/%s/raw`' % dataset)
        exit()
    parsed_path = os.path.join(dataset_path, 'parsed')
    
    if args.from_saved:
        patient_admission = pickle.load(open(os.path.join(parsed_path, 'patient_admission.pkl'), 'rb'))
        admission_codes = pickle.load(open(os.path.join(parsed_path, 'admission_codes.pkl'), 'rb'))
        admission_procedures = pickle.load(open(os.path.join(parsed_path, 'admission_procedures.pkl'), 'rb'))
    else:
        parser = PARSERS[dataset](raw_path)
        sample_num = args.sample_num if dataset == 'mimic4' else None
        patient_admission, admission_codes, admission_procedures = parser.parse(sample_num)
        print('saving parsed data ...')
        if not os.path.exists(parsed_path):
            os.makedirs(parsed_path)
        pickle.dump(patient_admission, open(os.path.join(parsed_path, 'patient_admission.pkl'), 'wb'))
        pickle.dump(admission_codes, open(os.path.join(parsed_path, 'admission_codes.pkl'), 'wb'))
        pickle.dump(admission_procedures, open(os.path.join(parsed_path, 'admission_procedures.pkl'), 'wb'))

    patient_num = len(patient_admission)

    def stat(data):
        if data is None:
            return 0, 0, 0
        lens = [len(item) for item in data.values()]
        max_, min_, avg = max(lens), min(lens), sum(lens) / len(data)
        return max_, min_, avg

    admission_stats = stat(patient_admission)
    visit_code_stats = stat(admission_codes)
    visit_procedure_stats = stat(admission_procedures)
    print('patient num: %d' % patient_num)
    print('visit num: %d' % len(admission_codes))
    print('procedure visit num: %d' % len(admission_procedures))
    print('max, min, mean admission num: %d, %d, %.2f' % admission_stats)
    print('max, min, mean visit code num: %d, %d, %.2f' % visit_code_stats)
    print('max, min, mean visit procedure num: %d, %d, %.2f' % visit_procedure_stats)

    max_admission_num = admission_stats[0]

    print('encoding codes ...')
    # SỬA ĐÚNG: encode cả diagnoses và procedures
    admission_diagnoses_encoded, diagnosis_map, admission_procedures_encoded, procedure_map = encode_concept(
        patient_admission, admission_codes, admission_procedures
    )
    diagnosis_num = len(diagnosis_map)
    procedure_num = len(procedure_map)
    total_codes = diagnosis_num + procedure_num
    print('There are %d diagnosis codes and %d procedure codes' % (diagnosis_num, procedure_num))
    print('Total codes: %d' % total_codes)
    
    # Generate code levels cho cả diagnoses và procedures
    print('generating code levels for diagnoses...')
    diagnosis_levels = generate_code_levels(data_path, diagnosis_map)
    print('generating code levels for procedures...')
    procedure_levels = generate_procedure_levels(data_path, procedure_map)
    pickle.dump({
        'diagnosis_levels': diagnosis_levels,
        'procedure_levels': procedure_levels
    }, open(os.path.join(parsed_path, 'diagnosis_levels.pkl'), 'wb'))

    print('splitting training, and test patients')
    # SỬA ĐÚNG: gọi hàm split_patients với parameters đúng
    train_pids, test_pids = split_patients(
        patient_admission=patient_admission,
        admission_diagnoses_encoded=admission_diagnoses_encoded,  # SỬA: dùng encoded data
        diagnosis_map=diagnosis_map,  # SỬA: thêm diagnosis_map
        admission_procedures_encoded=admission_procedures_encoded,  # SỬA: thêm procedures
        procedure_map=procedure_map,  # SỬA: thêm procedure_map
        train_num=args.train_num,
        seed=args.seed
    )
    print('There are %d train, %d test samples' % (len(train_pids), len(test_pids)))
    
    # SỬA ĐÚNG: generate adjacent matrix với parameters đúng
    code_adj = generate_code_code_adjacent(
        pids=train_pids, 
        patient_admission=patient_admission,
        admission_diagnoses_encoded=admission_diagnoses_encoded,
        diagnosis_num=diagnosis_num  # SỬA: chỉ cần diagnosis_num
    )

    # SỬA: common args cho dual streams
    common_args = [patient_admission, admission_diagnoses_encoded, admission_procedures_encoded, 
                   max_admission_num, diagnosis_num, procedure_num]

    print('build train real data ...')
    train_diagnoses, train_procedures, train_lens = build_real_data(train_pids, *common_args)
    print('build test real data ...')
    test_diagnoses, test_procedures, test_lens = build_real_data(test_pids, *common_args)


    assert train_diagnoses.shape[:2] == train_procedures.shape[:2], "Mismatch train diag/proc"
    assert test_diagnoses.shape[:2] == test_procedures.shape[:2], "Mismatch test diag/proc"


    # SỬA: statistics cho dual streams
    admission_dist, diag_visit_dist, diag_patient_dist, proc_visit_dist, proc_patient_dist = real_data_stat(
        train_diagnoses, train_procedures, train_lens
    )

    print('build train visit data ...')
    train_visit_x, train_timestep = build_visit_x(train_diagnoses, train_procedures, train_lens, diagnosis_num, procedure_num)

    print('build train real next ...')
    train_diag_next_x, train_diag_next_y, train_proc_next_x, train_proc_next_y, train_next_lens = build_real_next_xy(
        train_diagnoses, train_procedures, train_lens
    )

    print('building train codes features and labels ...')
    train_diag_x, train_diag_y, train_proc_x, train_proc_y, train_visit_lens = build_code_xy(
        train_diagnoses, train_procedures, train_lens
    )
    
    print('building test codes features and labels ...')
    test_diag_x, test_diag_y, test_proc_x, test_proc_y, test_visit_lens = build_code_xy(
        test_diagnoses, test_procedures, test_lens
    )

    # SỬA: heart failure chỉ trên diagnoses
    print('building train heart failure labels ...')
    train_hf_y = build_heart_failure_y('428', train_diag_y, diagnosis_map)
    print('building test heart failure labels ...')
    test_hf_y = build_heart_failure_y('428', test_diag_y, diagnosis_map)

    print('building train parkinson labels ...')
    train_parkinson_y = build_heart_failure_y('332', train_diag_y, diagnosis_map)
    print('building test parkinson labels ...')
    test_parkinson_y = build_heart_failure_y('332', test_diag_y, diagnosis_map)

    encoded_path = os.path.join(dataset_path, 'encoded')
    if not os.path.exists(encoded_path):
        os.makedirs(encoded_path)
    print('saving encoded data ...')
    pickle.dump(patient_admission, open(os.path.join(encoded_path, 'patient_admission.pkl'), 'wb'))
    pickle.dump(admission_diagnoses_encoded, open(os.path.join(encoded_path, 'diagnoses_encoded.pkl'), 'wb'))
    pickle.dump(admission_procedures_encoded, open(os.path.join(encoded_path, 'procedures_encoded.pkl'), 'wb'))
    pickle.dump(diagnosis_map, open(os.path.join(encoded_path, 'diagnosis_map.pkl'), 'wb'))
    pickle.dump(procedure_map, open(os.path.join(encoded_path, 'procedure_map.pkl'), 'wb'))
    pickle.dump({
        'train_pids': train_pids,
        'test_pids': test_pids
    }, open(os.path.join(encoded_path, 'pids.pkl'), 'wb'))
    
    # THÊM: save code numbers info
    pickle.dump({
        'diagnosis_num': diagnosis_num,
        'procedure_num': procedure_num,
        'total_codes': total_codes
    }, open(os.path.join(encoded_path, 'code_info.pkl'), 'wb'))

    print('saving standard data ...')
    standard_path = os.path.join(dataset_path, 'standard')
    if not os.path.exists(standard_path):
        os.makedirs(standard_path)

    # Ensure consistent dtype
    train_diagnoses = train_diagnoses.astype(np.float32)
    train_procedures = train_procedures.astype(np.float32)
    test_diagnoses = test_diagnoses.astype(np.float32)
    test_procedures = test_procedures.astype(np.float32)
    train_lens = train_lens.astype(np.int64)
    test_lens = test_lens.astype(np.int64)
    train_next_lens = train_next_lens.astype(np.int64)


    print('saving real data')
    real_data_path = os.path.join(standard_path, 'real_data')
    if not os.path.exists(real_data_path):
        os.makedirs(real_data_path)
    
    # TÁCH RIÊNG diagnoses và procedures
    print('\tsaving train diagnoses data ...')
    np.savez_compressed(os.path.join(real_data_path, 'train_diagnoses.npz'),
                        x=train_diagnoses, lens=train_lens)
    print('\tsaving train procedures data ...')
    np.savez_compressed(os.path.join(real_data_path, 'train_procedures.npz'),
                        x=train_procedures, lens=train_lens)
    print('\tsaving test diagnoses data ...')
    np.savez_compressed(os.path.join(real_data_path, 'test_diagnoses.npz'),
                        x=test_diagnoses, lens=test_lens)
    print('\tsaving test procedures data ...')
    np.savez_compressed(os.path.join(real_data_path, 'test_procedures.npz'),
                        x=test_procedures, lens=test_lens)

    print('saving visit data')
    visit_path = os.path.join(standard_path, 'single_visits')
    if not os.path.exists(visit_path):
        os.makedirs(visit_path)
    print('\tsaving train visit data ...')
    np.savez_compressed(os.path.join(visit_path, 'train.npz'), x=train_visit_x)
    np.savez_compressed(os.path.join(visit_path, 'train_timestep.npz'), x=train_timestep)

    print('saving real next data')
    real_next_path = os.path.join(standard_path, 'real_next')
    if not os.path.exists(real_next_path):
        os.makedirs(real_next_path)
    print('\tsaving train real next data ...')
    np.savez_compressed(os.path.join(real_next_path, 'train_diagnoses.npz'),
                        x=train_diag_next_x, lens=train_next_lens, y=train_diag_next_y)
    np.savez_compressed(os.path.join(real_next_path, 'train_procedures.npz'),
                        x=train_proc_next_x, lens=train_next_lens, y=train_proc_next_y)

    print('saving task data')
    task_path = os.path.join(standard_path, 'real_task')
    if not os.path.exists(task_path):
        os.makedirs(task_path)
        os.mkdir(os.path.join(task_path, 'train'))
        os.mkdir(os.path.join(task_path, 'test'))
    print('\tsaving task training data')
    np.savez_compressed(os.path.join(task_path, 'train', 'diagnoses_feature.npz'), x=train_diag_x, lens=train_visit_lens)
    np.savez_compressed(os.path.join(task_path, 'train', 'diagnoses_codes.npz'), y=train_diag_y)
    np.savez_compressed(os.path.join(task_path, 'train', 'procedures_feature.npz'), x=train_proc_x, lens=train_visit_lens)
    np.savez_compressed(os.path.join(task_path, 'train', 'procedures_codes.npz'), y=train_proc_y)
    np.savez_compressed(os.path.join(task_path, 'train', 'hf.npz'), y=train_hf_y)
    np.savez_compressed(os.path.join(task_path, 'train', 'parkinson.npz'), y=train_parkinson_y)
    
    print('\tsaving task test data')
    np.savez_compressed(os.path.join(task_path, 'test', 'diagnoses_feature.npz'), x=test_diag_x, lens=test_visit_lens)
    np.savez_compressed(os.path.join(task_path, 'test', 'diagnoses_codes.npz'), y=test_diag_y)
    np.savez_compressed(os.path.join(task_path, 'test', 'procedures_feature.npz'), x=test_proc_x, lens=test_visit_lens)
    np.savez_compressed(os.path.join(task_path, 'test', 'procedures_codes.npz'), y=test_proc_y)
    np.savez_compressed(os.path.join(task_path, 'test', 'hf.npz'), y=test_hf_y)
    np.savez_compressed(os.path.join(task_path, 'test', 'parkinson.npz'), y=test_parkinson_y)

    # SỬA saving statistics
    np.savez_compressed(os.path.join(standard_path, 'real_data_stat.npz'),
                        admission_dist=admission_dist,
                        diagnosis_visit_dist=diag_visit_dist,
                        diagnosis_patient_dist=diag_patient_dist,
                        procedure_visit_dist=proc_visit_dist,
                        procedure_patient_dist=proc_patient_dist)
    
    np.savez_compressed(os.path.join(standard_path, 'code_adj.npz'), code_adj=code_adj)