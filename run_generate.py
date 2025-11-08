import os
import random

import torch
import numpy as np

from config import get_generate_args, get_paths
from model import Generator
from datautils.dataloader import get_dual_name_maps, load_meta_data  # SỬA IMPORT
from datautils.dataset import DatasetReal
from generation.generate import generate_ehr, get_required_number
from generation.stat_ehr import get_basic_statistics, get_top_k_disease, calc_distance


def generate(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # SỬA: Lấy thêm diagnosis_num, procedure_num
    dataset_path, _, params_path, diagnosis_num, procedure_num, total_codes = get_paths(args)
    
    # SỬA: Load dual-stream metadata
    (len_dist, diag_visit_dist, diag_patient_dist, proc_visit_dist, proc_patient_dist, 
     code_adj, diagnosis_map, procedure_map, code_info) = load_meta_data(dataset_path)
    
    # SỬA QUAN TRỌNG: Load cả diagnosis và procedure name maps
    diagnosis_name_map, procedure_name_map = get_dual_name_maps(args.data_path)
    
    # SỬA: Tạo mapping cho dual-stream
    idiagnosis_map = {v: k for k, v in diagnosis_map.items()}
    iprocedure_map = {v: k for k, v in procedure_map.items()}

    # SỬA: Load real data - cần điều chỉnh DatasetReal cho dual-stream
    dataset_real = DatasetReal(os.path.join(dataset_path, 'standard', 'real_data'))
    len_dist = torch.from_numpy(len_dist).to(device)
    real_diagnoses, real_procedures, real_lens = dataset_real.train_set.data
    max_len = real_diagnoses.shape[1]

    

    if args.use_iteration == -1:
        param_file_name = 'generator.pt'
    else:
        param_file_name = 'generator.{}.pt'.format(args.use_iteration)

    # SỬA: Khởi tạo dual-stream Generator
    generator = Generator(
        diagnosis_num=diagnosis_num,
        procedure_num=procedure_num,
        hidden_dim=args.g_hidden_dim,
        attention_dim=args.g_attention_dim,
        max_len=max_len,
        device=device
    ).to(device)
    generator.load(params_path, param_file_name)

    # SỬA: Generate dual-stream data
    fake_diagnoses, fake_procedures, fake_lens = generate_ehr(generator, args.number, len_dist, args.batch_size)

    """------------------------get statistics------------------------"""
    print('=== REAL DATA ===')
    # Statistics cho diagnosis
    diag_n_types, diag_n_codes, diag_n_visits, diag_avg_code_num, diag_avg_visit_num = get_basic_statistics(
        real_diagnoses, real_lens
    )
    # Statistics cho procedure  
    proc_n_types, proc_n_codes, proc_n_visits, proc_avg_code_num, proc_avg_visit_num = get_basic_statistics(
        real_procedures, real_lens
    )
    
    print('{} samples -- Diagnosis: {} types, {} codes, avg {:.4f} codes/visit | Procedure: {} types, {} codes, avg {:.4f} codes/visit | Avg visits: {:.4f}'
          .format(len(real_diagnoses), diag_n_types, diag_n_codes, diag_avg_code_num, 
                 proc_n_types, proc_n_codes, proc_avg_code_num, diag_avg_visit_num))
    
    print('\n--- TOP 10 DIAGNOSES ---')
    get_top_k_disease(real_diagnoses, real_lens, idiagnosis_map, diagnosis_name_map, top_k=10)
    
    print('\n--- TOP 10 PROCEDURES ---')
    get_top_k_disease(real_procedures, real_lens, iprocedure_map, procedure_name_map, top_k=10)

    print('\n=== SYNTHETIC DATA ===')
    # Statistics cho fake diagnosis
    fake_diag_n_types, fake_diag_n_codes, fake_diag_n_visits, fake_diag_avg_code_num, fake_diag_avg_visit_num = get_basic_statistics(
        fake_diagnoses, fake_lens
    )
    # Statistics cho fake procedure  
    fake_proc_n_types, fake_proc_n_codes, fake_proc_n_visits, fake_proc_avg_code_num, fake_proc_avg_visit_num = get_basic_statistics(
        fake_procedures, fake_lens
    )
    
    print('{} samples -- Diagnosis: {} types, {} codes, avg {:.4f} codes/visit | Procedure: {} types, {} codes, avg {:.4f} codes/visit | Avg visits: {:.4f}'
          .format(args.number, fake_diag_n_types, fake_diag_n_codes, fake_diag_avg_code_num, 
                 fake_proc_n_types, fake_proc_n_codes, fake_proc_avg_code_num, fake_diag_avg_visit_num))
    
    print('\n--- TOP 10 DIAGNOSES ---')
    get_top_k_disease(fake_diagnoses, fake_lens, idiagnosis_map, diagnosis_name_map, top_k=10)  # SỬA
    
    print('\n--- TOP 10 PROCEDURES ---')
    get_top_k_disease(fake_procedures, fake_lens, iprocedure_map, procedure_name_map, top_k=10)  # SỬA

    # SỬA: Tính distance cho cả diagnosis và procedure
    print('\n=== DISTANCE METRICS ===')
    # Diagnosis distance
    diag_jsd_v, diag_jsd_p, diag_nd_v, diag_nd_p = calc_distance(
        real_diagnoses, real_lens,
        fake_diagnoses, fake_lens,
        diagnosis_num
    )
    print('Diagnosis - JSD_v: {:.4f}, JSD_p: {:.4f}, ND_v: {:.4f}, ND_p: {:.4f}'.format(diag_jsd_v, diag_jsd_p, diag_nd_v, diag_nd_p))
    
    # Procedure distance  
    proc_jsd_v, proc_jsd_p, proc_nd_v, proc_nd_p = calc_distance(
        real_procedures, real_lens,
        fake_procedures, fake_lens,
        procedure_num
    )
    print('Procedure - JSD_v: {:.4f}, JSD_p: {:.4f}, ND_v: {:.4f}, ND_p: {:.4f}'.format(proc_jsd_v, proc_jsd_p, proc_nd_v, proc_nd_p))
    """------------------------get statistics------------------------"""

    get_required_number(generator, len_dist, args.batch_size, args.upper_bound)

    print('saving {} synthetic dual-stream data...'.format(args.number))
    synthetic_path = os.path.join(args.result_path, 'synthetic_{}_dual.npz'.format(args.dataset))
    # SỬA: Lưu cả diagnoses và procedures
    np.savez_compressed(synthetic_path, 
                       diagnoses=fake_diagnoses, 
                       procedures=fake_procedures, 
                       lens=fake_lens)


if __name__ == '__main__':
    args = get_generate_args()
    generate(args)