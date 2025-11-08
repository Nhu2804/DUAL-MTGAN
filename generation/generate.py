import torch
import numpy as np


def generate_ehr(generator, number, len_dist, batch_size):
    """Generate synthetic EHR data for dual-stream model"""
    fake_diagnoses, fake_procedures, fake_lens = [], [], []
    
    for i in range(0, number, batch_size):
        n = number - i if i + batch_size > number else batch_size
        
        # Lấy target codes cho cả diagnosis và procedure
        target_diagnoses, target_procedures = generator.get_target_codes(n, code_type='both')
        
        # Tạo lens - SỬA WARNING: dùng detach().clone()
        len_dist_tensor = torch.as_tensor(len_dist, dtype=torch.float32).detach()
        lens = torch.multinomial(len_dist_tensor, num_samples=n, replacement=True) + 1
        lens = lens.to(generator.device)
        
        # Generate synthetic data
        x_diagnoses, x_procedures = generator.sample(target_diagnoses, target_procedures, lens)

        fake_diagnoses.append(x_diagnoses.cpu().numpy())
        fake_procedures.append(x_procedures.cpu().numpy())
        fake_lens.append(lens.cpu().numpy())
    
    fake_diagnoses = np.concatenate(fake_diagnoses, axis=0)
    fake_procedures = np.concatenate(fake_procedures, axis=0)
    fake_lens = np.concatenate(fake_lens, axis=-1)

    return fake_diagnoses, fake_procedures, fake_lens


def get_required_number(generator, len_dist, batch_size, upper_bound=1e7):
    """Calculate required number to generate all code types for dual-stream"""
    # SỬA: Dùng generator.diagnosis_num và generator.procedure_num
    diag_code_types = torch.zeros(generator.diagnosis_num, dtype=torch.bool, device=generator.device)
    proc_code_types = torch.zeros(generator.procedure_num, dtype=torch.bool, device=generator.device)
    
    rn = 0
    while True:
        n = np.random.randint(low=np.floor(0.5 * batch_size), high=np.floor(1.5 * batch_size))
        rn += n

        # Lấy target codes và lens
        target_diagnoses, target_procedures = generator.get_target_codes(n, code_type='both')
        
        # SỬA WARNING: dùng detach().clone()
        len_dist_tensor = torch.as_tensor(len_dist, dtype=torch.float32).detach()
        lens = torch.multinomial(len_dist_tensor, num_samples=n, replacement=True) + 1
        lens = lens.to(generator.device)
        
        # Generate samples
        x_diagnoses, x_procedures = generator.sample(target_diagnoses, target_procedures, lens)

        # Update code coverage - SỬA CÁCH TÍNH
        # Sửa: sum(dim=1).sum(dim=0) thành sum(dim=(0,1)) cho đúng
        diag_code_types = torch.logical_or(diag_code_types, (x_diagnoses.sum(dim=(0, 1)) > 0))
        proc_code_types = torch.logical_or(proc_code_types, (x_procedures.sum(dim=(0, 1)) > 0))
        
        total_diag_types = diag_code_types.sum()
        total_proc_types = proc_code_types.sum()
        
        print(f'Diagnosis codes: {total_diag_types.item()}/{generator.diagnosis_num}, '
              f'Procedure codes: {total_proc_types.item()}/{generator.procedure_num}, '
              f'Samples: {rn}')
        
        # Kiểm tra điều kiện dừng
        if total_diag_types == generator.diagnosis_num and total_proc_types == generator.procedure_num:
            print(f'Required number to generate all codes: {rn}')
            return rn
            
        if rn >= upper_bound:
            print(f'Unable to generate all codes within {upper_bound} samples')
            print(f'Coverage: Diagnosis {total_diag_types.item()}/{generator.diagnosis_num}, '
                  f'Procedure {total_proc_types.item()}/{generator.procedure_num}')
            return rn