import numpy as np

from preprocess.parse_csv import EHRParser


def generate_code_code_adjacent(pids, patient_admission, admission_diagnoses_encoded, diagnosis_num):
    """GIỮ NGUYÊN LOGIC GỐC: chỉ xử lý diagnosis codes cho adjacent matrix"""
    print('generating code code adjacent matrix ...')
    adj = np.zeros((diagnosis_num, diagnosis_num), dtype=int)
    
    for i, pid in enumerate(pids):
        print('\r\t%d / %d' % (i + 1, len(pids)), end='')
        for admission in patient_admission[pid]:
            adm_id = admission[EHRParser.adm_id_col]
            
            # CHỈ xử lý diagnoses (như gốc)
            if adm_id in admission_diagnoses_encoded:
                codes = admission_diagnoses_encoded[adm_id]
                
                # Tạo adjacent matrix (giữ nguyên logic gốc)
                if len(codes) >= 2:
                    for row in range(len(codes) - 1):
                        for col in range(row + 1, len(codes)):
                            c_i = codes[row]
                            c_j = codes[col]
                            if c_i < diagnosis_num and c_j < diagnosis_num:
                                adj[c_i, c_j] = 1
                                adj[c_j, c_i] = 1
    print('\r\t%d / %d' % (len(pids), len(pids)))
    return adj


def real_data_stat(real_diagnoses, real_procedures, lens):
    admission_num_count = {}
    max_admission_num = 0
    
    # Thống kê RIÊNG cho diagnoses
    diagnosis_visit_count = {}
    diagnosis_patient_count = {}
    
    # Thống kê RIÊNG cho procedures  
    procedure_visit_count = {}
    procedure_patient_count = {}
    
    for diag_patient, proc_patient, len_i in zip(real_diagnoses, real_procedures, lens):
        if max_admission_num < len_i:
            max_admission_num = len_i
        admission_num_count[len_i] = admission_num_count.get(len_i, 0) + 1

        diagnosis_set = set()
        procedure_set = set()
        
        for i in range(len_i):
            # Diagnoses statistics
            diagnosis_codes = np.where(diag_patient[i] > 0)[0]
            diagnosis_set.update(diagnosis_codes.tolist())
            for code in diagnosis_codes:
                diagnosis_visit_count[code] = diagnosis_visit_count.get(code, 0) + 1
            
            # Procedures statistics
            procedure_codes = np.where(proc_patient[i] > 0)[0]
            procedure_set.update(procedure_codes.tolist())
            for code in procedure_codes:
                procedure_visit_count[code] = procedure_visit_count.get(code, 0) + 1

        # Patient-level counts
        for code in diagnosis_set:
            diagnosis_patient_count[code] = diagnosis_patient_count.get(code, 0) + 1
        for code in procedure_set:
            procedure_patient_count[code] = procedure_patient_count.get(code, 0) + 1

    # Admission distribution
    admission_dist = np.zeros((max_admission_num, ))
    for num, count in admission_num_count.items():
        if num - 1 < len(admission_dist):  # THÊM KIỂM TRA
            admission_dist[num - 1] = count
    if admission_dist.sum() > 0:
        admission_dist /= admission_dist.sum()

    # SỬA: XỬ LÝ TRƯỜNG HỢP RỖNG
    max_diagnosis_code = max(diagnosis_visit_count.keys()) if diagnosis_visit_count else -1
    max_procedure_code = max(procedure_visit_count.keys()) if procedure_visit_count else -1

    # Diagnosis distributions - SỬA: XỬ LÝ KHI KHÔNG CÓ DATA
    if max_diagnosis_code >= 0:
        diagnosis_visit_dist = np.zeros(max_diagnosis_code + 1)
        for code, count in diagnosis_visit_count.items():
            if code < len(diagnosis_visit_dist):  # THÊM KIỂM TRA
                diagnosis_visit_dist[code] = count
        if diagnosis_visit_dist.sum() > 0:
            diagnosis_visit_dist /= diagnosis_visit_dist.sum()
    else:
        diagnosis_visit_dist = np.zeros(0)

    if max_diagnosis_code >= 0:
        diagnosis_patient_dist = np.zeros(max_diagnosis_code + 1)
        for code, count in diagnosis_patient_count.items():
            if code < len(diagnosis_patient_dist):
                diagnosis_patient_dist[code] = count
        if diagnosis_patient_dist.sum() > 0:
            diagnosis_patient_dist /= diagnosis_patient_dist.sum()
    else:
        diagnosis_patient_dist = np.zeros(0)

    # Procedure distributions - SỬA: XỬ LÝ KHI KHÔNG CÓ DATA
    if max_procedure_code >= 0:
        procedure_visit_dist = np.zeros(max_procedure_code + 1)
        for code, count in procedure_visit_count.items():
            if code < len(procedure_visit_dist):
                procedure_visit_dist[code] = count
        if procedure_visit_dist.sum() > 0:
            procedure_visit_dist /= procedure_visit_dist.sum()
    else:
        procedure_visit_dist = np.zeros(0)

    if max_procedure_code >= 0:
        procedure_patient_dist = np.zeros(max_procedure_code + 1)
        for code, count in procedure_patient_count.items():
            if code < len(procedure_patient_dist):
                procedure_patient_dist[code] = count
        if procedure_patient_dist.sum() > 0:
            procedure_patient_dist /= procedure_patient_dist.sum()
    else:
        procedure_patient_dist = np.zeros(0)

    return (admission_dist, 
            diagnosis_visit_dist, diagnosis_patient_dist,
            procedure_visit_dist, procedure_patient_dist)


def parse_icd9_range(range_: str):
    ranges = range_.lstrip().split('-')
    if ranges[0][0] == 'V':
        prefix = 'V'
        format_ = '%02d'
        start, end = int(ranges[0][1:]), int(ranges[1][1:])
    elif ranges[0][0] == 'E':
        prefix = 'E'
        format_ = '%03d'
        start, end = int(ranges[0][1:]), int(ranges[1][1:])
    else:
        prefix = ''
        format_ = '%03d'
        if len(ranges) == 1:
            start = int(ranges[0])
            end = start
        else:
            start, end = int(ranges[0]), int(ranges[1])
    return prefix, format_, start, end


def to_standard_icd9(code: str):
        code = str(code)
        if code == '':
            return code
        split_pos = 4 if code.startswith('E') else 3
        icd9_code = code[:split_pos] + '.' + code[split_pos:] if len(code) > split_pos else code
        return icd9_code


def generate_code_levels(path, code_map: dict) -> np.ndarray:
    print('generating code levels ...')
    import os
    code_map = {to_standard_icd9(code): cid for code, cid in code_map.items()}
    three_level_code_set = set(code.split('.')[0] for code in code_map)
    icd9_path = os.path.join(path, 'icd9.txt')
    icd9_range = list(open(icd9_path, 'r', encoding='utf-8').readlines())
    three_level_dict = dict()
    level1, level2, level3 = (0, 0, 0)
    level1_can_add = False
    for range_ in icd9_range:
        range_ = range_.rstrip()
        if range_[0] == ' ':
            prefix, format_, start, end = parse_icd9_range(range_)
            level2_cannot_add = True
            for i in range(start, end + 1):
                code = prefix + format_ % i
                if code in three_level_code_set:
                    three_level_dict[code] = [level1, level2, level3]
                    level3 += 1
                    level1_can_add = True
                    level2_cannot_add = False
            if not level2_cannot_add:
                level2 += 1
        else:
            if level1_can_add:
                level1 += 1
                level1_can_add = False

    code_level = dict()
    for code, cid in code_map.items():
        three_level_code = code.split('.')[0]
        three_level = three_level_dict.get(three_level_code, [0, 0, 0])  # SỬA: DÙNG GET ĐỂ TRÁNH KEYERROR
        code_level[code] = three_level + [cid]

    code_level_matrix = np.zeros((len(code_map), 4), dtype=int)
    for code, cid in code_map.items():
        code_level_matrix[cid] = code_level[code]

    return code_level_matrix


def generate_procedure_levels(path, procedure_map: dict) -> np.ndarray:
    """Generate hierarchy levels for procedure codes"""
    print('generating procedure levels ...')
    # Procedure codes thường không có hierarchy phức tạp như diagnoses
    # Tạo simple hierarchy dựa trên code prefix
    procedure_level_matrix = np.zeros((len(procedure_map), 4), dtype=int)
    
    for procedure, pid in procedure_map.items():
        # Giả sử 2 chữ số đầu là category level
        try:
            category = int(procedure[:2]) if procedure[:2].isdigit() else 0
        except:
            category = 0
        
        procedure_level_matrix[pid] = [category // 10, category % 10, 0, pid]
    
    return procedure_level_matrix