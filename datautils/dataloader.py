import os
import pickle

import numpy as np
import pandas as pd

from .dataset import DatasetReal, DatasetRealNext


def infinite_dataloader(dataloader):
    while True:
        for x in dataloader:
            yield x


class DataLoader:
    def __init__(self, dataset, shuffle=True, batch_size=32):
        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size

        self.size = len(dataset)
        self.idx = np.arange(self.size)
        # SỬA: Đảm bảo luôn có ít nhất 1 batch, đặc biệt cho test loader
        self.n_batches = max(1, np.ceil(self.size / batch_size).astype(int))

        self.counter = 0
        if shuffle:
            np.random.shuffle(self.idx)

    def _get_item(self, index):
        start = index * self.batch_size
        end = start + self.batch_size
        index = self.idx[start:end]
        data = self.dataset[index]
        return data

    def __next__(self):
        if self.counter >= self.n_batches:
            self.counter = 0
            if self.shuffle:
                np.random.shuffle(self.idx)
            raise StopIteration
        data = self._get_item(self.counter)
        self.counter += 1
        return data

    def __iter__(self):
        return self

    def __len__(self):
        return self.n_batches


def get_train_test_loader(dataset_path, batch_size, device):
    dataset = DatasetReal(os.path.join(dataset_path, 'standard', 'real_data'), device=device)
    
    # SỬA: Dùng batch size nhỏ hơn cho test để tránh division by zero
    train_loader = DataLoader(dataset.train_set, shuffle=True, batch_size=batch_size)
    test_batch_size = min(32, batch_size)  # Đảm bảo test loader không rỗng
    test_loader = DataLoader(dataset.test_set, shuffle=False, batch_size=test_batch_size)
    
    # SỬA: Lấy max_len và thống kê cho dual-stream
    max_len = dataset.train_set.data[0].shape[1]

    # SỬA: Thống kê riêng cho diagnoses và procedures
    print('total diagnosis codes in train:', dataset.train_set.data[0].sum())
    print('total procedure codes in train:', dataset.train_set.data[1].sum())
    print('total diagnosis codes in test:', dataset.test_set.data[0].sum())
    print('total procedure codes in test:', dataset.test_set.data[1].sum())
    
    # DEBUG: In kích thước datasets
    print(f"Train dataset size: {len(dataset.train_set)}")
    print(f"Test dataset size: {len(dataset.test_set)}")
    print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
    
    return train_loader, test_loader, max_len


def get_base_gru_train_loader(dataset_path, batch_size, device):
    dataset = DatasetRealNext(os.path.join(dataset_path, 'standard', 'real_next'), device=device)
    train_loader = DataLoader(dataset.train_set, shuffle=True, batch_size=batch_size)
    return train_loader


def load_meta_data(dataset_path):
    standard_path = os.path.join(dataset_path, 'standard')
    encoded_path = os.path.join(dataset_path, 'encoded')
    
    # SỬA: Load statistics cho dual streams
    real_data_stat = np.load(os.path.join(standard_path, 'real_data_stat.npz'))
    len_dist = real_data_stat['admission_dist']
    
    # SỬA: Tách riêng diagnosis và procedure distributions
    diag_visit_dist = real_data_stat['diagnosis_visit_dist']
    diag_patient_dist = real_data_stat['diagnosis_patient_dist']
    proc_visit_dist = real_data_stat['procedure_visit_dist']
    proc_patient_dist = real_data_stat['procedure_patient_dist']
    
    code_adj = np.load(os.path.join(standard_path, 'code_adj.npz'))['code_adj']
    
    # SỬA: Load cả diagnosis_map và procedure_map
    diagnosis_map = pickle.load(open(os.path.join(encoded_path, 'diagnosis_map.pkl'), 'rb'))
    procedure_map = pickle.load(open(os.path.join(encoded_path, 'procedure_map.pkl'), 'rb'))
    code_info = pickle.load(open(os.path.join(encoded_path, 'code_info.pkl'), 'rb'))
    
    return (len_dist, diag_visit_dist, diag_patient_dist, proc_visit_dist, proc_patient_dist, 
            code_adj, diagnosis_map, procedure_map, code_info)


def load_diagnosis_name_map(data_path):
    """Load diagnosis names từ map.xlsx"""
    names = pd.read_excel(os.path.join(data_path, 'map.xlsx'), engine='openpyxl')
    code_keys = names['DIAGNOSIS CODE'].tolist()
    name_vals = names['LONG DESCRIPTION'].tolist()
    diagnosis_name_map = {k: v for k, v in zip(code_keys, name_vals)}
    print(f"Loaded {len(diagnosis_name_map)} diagnosis names")
    return diagnosis_name_map


def load_procedure_name_map(data_path):
    """Load procedure names từ map_procedure.xlsx"""
    try:
        names = pd.read_excel(os.path.join(data_path, 'map_procedure.xlsx'), engine='openpyxl')
        # Kiểm tra cấu trúc file
        if 'PROCEDURE CODE' in names.columns:
            code_keys = names['PROCEDURE CODE'].tolist()
        elif 'CODE' in names.columns:
            code_keys = names['CODE'].tolist()
        else:
            code_keys = names.iloc[:, 0].tolist()
        
        if 'LONG DESCRIPTION' in names.columns:
            name_vals = names['LONG DESCRIPTION'].tolist()
        elif 'DESCRIPTION' in names.columns:
            name_vals = names['DESCRIPTION'].tolist()
        else:
            name_vals = names.iloc[:, 1].tolist()
        
        procedure_name_map = {str(k): v for k, v in zip(code_keys, name_vals)}
        print(f"Loaded {len(procedure_name_map)} procedure names")
        return procedure_name_map
    except FileNotFoundError:
        print("⚠️  map_procedure.xlsx not found, using code-only display for procedures")
        return {}
    except Exception as e:
        print(f"⚠️  Error loading procedure names: {e}")
        return {}


def get_dual_name_maps(data_path):
    """Load cả diagnosis và procedure name maps"""
    diagnosis_name_map = load_diagnosis_name_map(data_path)
    procedure_name_map = load_procedure_name_map(data_path)
    return diagnosis_name_map, procedure_name_map


def get_code_numbers(dataset_path):
    """Lấy số lượng diagnoses và procedures"""
    encoded_path = os.path.join(dataset_path, 'encoded')
    code_info = pickle.load(open(os.path.join(encoded_path, 'code_info.pkl'), 'rb'))
    diagnosis_num = code_info['diagnosis_num']
    procedure_num = code_info['procedure_num']
    total_codes = code_info['total_codes']
    return diagnosis_num, procedure_num, total_codes