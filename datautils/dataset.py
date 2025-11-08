import os
import torch
import numpy as np


# ============================================================
# 🧩 Dataset cơ bản
# ============================================================
class Dataset:
    def __init__(self, inputs, device):
        self.data = inputs
        self.device = device
        self.size = len(inputs[0])  # inputs = (diag_x, proc_x, lens)

    def __len__(self):
        return self.size

    def __getitem__(self, indices):
        # Trả về tensor của cả diagnosis, procedure, và lens
        data = [torch.tensor(x[indices], device=self.device) for x in self.data]
        return data


# ============================================================
# 🧩 DatasetReal (Dual-stream: Diagnosis + Procedure)
# ============================================================
class DatasetReal:
    def __init__(self, path, device=None):
        self.path = path
        self.device = device
        print('loading real dual-stream data ...')
        print('\tloading real training data ...')
        self.train_set = self._load('train_diagnoses.npz', 'train_procedures.npz')
        print('\tloading real test data ...')
        self.test_set = self._load('test_diagnoses.npz', 'test_procedures.npz')

    def _load(self, diag_file, proc_file):
        # 1️⃣ Load diagnosis và procedure song song
        diag_data = np.load(os.path.join(self.path, diag_file))
        proc_data = np.load(os.path.join(self.path, proc_file))

        diag_x = diag_data['x'].astype(np.float32)
        proc_x = proc_data['x'].astype(np.float32)
        lens = diag_data['lens'].astype(np.int64)

        # 2️⃣ Căn chỉnh số bệnh nhân (nếu khác nhau)
        min_len = min(len(diag_x), len(proc_x))
        diag_x, proc_x, lens = diag_x[:min_len], proc_x[:min_len], lens[:min_len]

        # 3️⃣ Giữ bệnh nhân không có procedure (thay bằng vector 0)
        for i in range(len(proc_x)):
            if np.all(proc_x[i] == 0):
                proc_x[i] = np.zeros_like(proc_x[0])

        # 4️⃣ Đồng bộ chiều dài chuỗi visit giữa diag & proc
        max_len = max(diag_x.shape[1], proc_x.shape[1])
        if diag_x.shape[1] < max_len:
            pad_len = max_len - diag_x.shape[1]
            diag_x = np.pad(diag_x, ((0, 0), (0, pad_len), (0, 0)), 'constant')
        if proc_x.shape[1] < max_len:
            pad_len = max_len - proc_x.shape[1]
            proc_x = np.pad(proc_x, ((0, 0), (0, pad_len), (0, 0)), 'constant')

        # 5️⃣ Trả về dataset chứa cả diag + proc
        dataset = Dataset((diag_x, proc_x, lens), self.device)
        return dataset


# ============================================================
# 🧩 DatasetRealNext (nếu dùng cho dự đoán next visit)
# ============================================================
class DatasetRealNext:
    def __init__(self, path, device=None):
        self.path = path
        self.device = device
        print('loading real next dual-stream data ...')
        print('\tloading real next training data ...')
        self.train_set = self._load('train')

    def _load(self, split):
        # Load diagnosis next data
        diag_data = np.load(os.path.join(self.path, f'{split}_diagnoses.npz'))
        diag_x = diag_data['x'].astype(np.float32)
        diag_lens = diag_data['lens'].astype(np.int64)
        diag_y = diag_data['y'].astype(np.float32)

        # Load procedure next data
        proc_data = np.load(os.path.join(self.path, f'{split}_procedures.npz'))
        proc_x = proc_data['x'].astype(np.float32)
        proc_lens = proc_data['lens'].astype(np.int64)
        proc_y = proc_data['y'].astype(np.float32)

        # Check alignment
        assert len(diag_x) == len(proc_x), "Mismatch between diagnosis and procedure samples"
        assert diag_x.shape[1] == proc_x.shape[1], "Mismatch in sequence length"

        dataset = Dataset(
            (diag_x, diag_lens, diag_y,
             proc_x, proc_lens, proc_y),
            self.device
        )
        return dataset
