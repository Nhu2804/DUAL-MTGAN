import numpy as np

from preprocess.parse_csv import EHRParser


def split_patients(patient_admission, admission_diagnoses_encoded, diagnosis_map, admission_procedures_encoded, procedure_map, train_num, seed=6669):
    """Dual-stream version: consider both diagnoses and procedures for splitting"""
    np.random.seed(seed)
    common_pids = set()
    
    # Check diagnoses coverage
    for i, code in enumerate(diagnosis_map):
        print('\r\tDiagnoses: %.2f%%' % ((i + 1) * 100 / len(diagnosis_map)), end='')
        for pid, admissions in patient_admission.items():
            for admission in admissions:
                adm_id = admission[EHRParser.adm_id_col]
                if adm_id in admission_diagnoses_encoded:
                    codes = admission_diagnoses_encoded[adm_id]
                    if code in codes:
                        common_pids.add(pid)
                        break
            else:
                continue
            break
    
    # Check procedures coverage  
    for i, code in enumerate(procedure_map):
        print('\r\tProcedures: %.2f%%' % ((i + 1) * 100 / len(procedure_map)), end='')
        for pid, admissions in patient_admission.items():
            for admission in admissions:
                adm_id = admission[EHRParser.adm_id_col]
                if adm_id in admission_procedures_encoded:
                    codes = admission_procedures_encoded[adm_id]
                    if code in codes:
                        common_pids.add(pid)
                        break
            else:
                continue
            break
    
    print('\r\t100%')
    
    # Ensure we have patients with max admissions
    max_admission_num = 0
    pid_max_admission_num = 0
    for pid, admissions in patient_admission.items():
        if len(admissions) > max_admission_num:
            max_admission_num = len(admissions)
            pid_max_admission_num = pid
    common_pids.add(pid_max_admission_num)
    
    remaining_pids = np.array(list(set(patient_admission.keys()).difference(common_pids)))
    np.random.shuffle(remaining_pids)

    train_pids = np.array(list(common_pids.union(set(remaining_pids[:(train_num - len(common_pids))].tolist()))))
    test_pids = remaining_pids[(train_num - len(common_pids)):]
    
    print(f"Training patients: {len(train_pids)}, Test patients: {len(test_pids)}")
    return train_pids, test_pids


def build_real_data(pids, patient_admission, admission_diagnoses_encoded, admission_procedures_encoded, max_admission_num, diagnosis_num, procedure_num):
    n = len(pids)
    # Tạo RIÊNG diagnoses và procedures
    x_diagnoses = np.zeros((n, max_admission_num, diagnosis_num), dtype=bool)
    x_procedures = np.zeros((n, max_admission_num, procedure_num), dtype=bool)
    lens = np.zeros((n, ), dtype=int)
    
    for i, pid in enumerate(pids):
        print('\r\t%d / %d' % (i + 1, len(pids)), end='')
        admissions = patient_admission[pid]
        for k, admission in enumerate(admissions):
            adm_id = admission[EHRParser.adm_id_col]
            
            # Diagnoses
            if adm_id in admission_diagnoses_encoded:
                diagnosis_codes = admission_diagnoses_encoded[adm_id]
                x_diagnoses[i, k, diagnosis_codes] = 1
            
            # Procedures  
            if adm_id in admission_procedures_encoded:
                procedure_codes = admission_procedures_encoded[adm_id]
                x_procedures[i, k, procedure_codes] = 1
                
        lens[i] = len(admissions)
    print('\r\t%d / %d' % (len(pids), len(pids)))
    
    return x_diagnoses, x_procedures, lens


def build_code_xy(real_diagnoses, real_procedures, real_lens):
    # Xử lý riêng diagnoses
    x_diagnoses = np.zeros_like(real_diagnoses)
    y_diagnoses = np.zeros((real_diagnoses.shape[0], real_diagnoses.shape[2]), dtype=bool)
    
    # Xử lý riêng procedures
    x_procedures = np.zeros_like(real_procedures)
    y_procedures = np.zeros((real_procedures.shape[0], real_procedures.shape[2]), dtype=bool)
    
    lens = real_lens - 1
    
    for i, (diag_i, proc_i, len_i) in enumerate(zip(real_diagnoses, real_procedures, lens)):
        # Diagnoses
        x_diagnoses[i][:len_i] = diag_i[:len_i]
        y_diagnoses[i] = diag_i[len_i]
        
        # Procedures
        x_procedures[i][:len_i] = proc_i[:len_i]
        y_procedures[i] = proc_i[len_i]
    
    return x_diagnoses, y_diagnoses, x_procedures, y_procedures, lens


def build_visit_x(diagnoses_data, procedures_data, lens, diagnosis_num, procedure_num):
    n = np.sum(lens)
    # Tạo combined visit data (nếu cần)
    x_combined = np.zeros((n, diagnosis_num + procedure_num), dtype=bool)
    t = np.zeros((n, ), dtype=int)
    
    i = 0
    for diag_patient, proc_patient, len_ in zip(diagnoses_data, procedures_data, lens):
        for k in range(len_):
            # Kết hợp diagnoses và procedures
            x_combined[i] = np.concatenate([diag_patient[k], proc_patient[k]])
            t[i] = k
            i += 1
    return x_combined, t


def build_real_next_xy(real_diagnoses, real_procedures, real_lens):
    x_diagnoses = np.zeros_like(real_diagnoses)
    y_diagnoses = np.zeros_like(real_diagnoses)
    x_procedures = np.zeros_like(real_procedures)
    y_procedures = np.zeros_like(real_procedures)
    
    lens = real_lens - 1
    
    for i, (diag_i, proc_i, len_i) in enumerate(zip(real_diagnoses, real_procedures, lens)):
        # Diagnoses
        x_diagnoses[i][:len_i] = diag_i[:len_i]
        y_diagnoses[i][:len_i] = diag_i[1:(len_i + 1)]
        
        # Procedures
        x_procedures[i][:len_i] = proc_i[:len_i]
        y_procedures[i][:len_i] = proc_i[1:(len_i + 1)]
    
    return x_diagnoses, y_diagnoses, x_procedures, y_procedures, lens


def build_heart_failure_y(hf_prefix, diagnosis_y, diagnosis_map):
    """CHỈ xử lý diagnoses (heart failure là diagnosis, không phải procedure)"""
    # SỬA: diagnosis_map là {code: index}, nên cần iterate đúng
    hf_list = []
    for code, idx in diagnosis_map.items():
        if code.startswith(hf_prefix):
            hf_list.append(idx)
    
    hf_list = np.array(hf_list)
    hfs = np.zeros((diagnosis_y.shape[1],), dtype=int)
    hfs[hf_list] = 1
    hf_exist = np.logical_and(diagnosis_y, hfs)
    y = (np.sum(hf_exist, axis=-1) > 0).astype(bool)
    return y