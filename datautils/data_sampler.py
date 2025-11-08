import torch
import numpy as np


class CodeSampleIter:
    def __init__(self, code, samples, shuffle=True):
        self.code = code
        self.samples = samples
        self.current_index = 0
        self.length = len(samples)
        self.shuffle = shuffle
        if self.shuffle:
            np.random.shuffle(self.samples)

    def __next__(self):
        sample = self.samples[self.current_index]
        self.current_index += 1
        if self.current_index == self.length:
            self.current_index = 0
            if self.shuffle:
                np.random.shuffle(self.samples)
        return sample


class DataSampler:
    def __init__(self, diagnoses_data, procedures_data, lens, device=None):
        self.diagnoses_data = diagnoses_data
        self.procedures_data = procedures_data
        self.lens = lens
        self.device = device
        self.size = len(diagnoses_data)

        # Mapping riêng cho diagnosis / procedure
        self.diagnosis_code_samples = self._get_code_sample_map(diagnoses_data, 'diagnosis')
        self.procedure_code_samples = self._get_code_sample_map(procedures_data, 'procedure')

    def _get_code_sample_map(self, data, code_type):
        print(f'building {code_type} data sampler ...')
        code_sample_map = dict()
        for i, (sample, len_i) in enumerate(zip(data, self.lens)):
            for t in range(len_i):
                visit = sample[t]
                codes = np.where(visit > 0)[0]
                for code in codes:
                    if code not in code_sample_map:
                        code_sample_map[code] = {i}
                    else:
                        code_sample_map[code].add(i)

        if len(code_sample_map) == 0:
            raise ValueError(f"No codes found in {code_type} data.")

        code_samples = [None] * (max(code_sample_map.keys()) + 1)
        for code, samples in code_sample_map.items():
            code_samples[code] = CodeSampleIter(code, list(samples))
        return code_samples

    def sample(self, target_diagnoses=None, target_procedures=None):
        # Giống MTGAN gốc: không ép batch size
        if target_diagnoses is not None:
            lines = [next(self.diagnosis_code_samples[code])
                     for code in target_diagnoses
                     if self.diagnosis_code_samples[code] is not None]
        elif target_procedures is not None:
            lines = [next(self.procedure_code_samples[code])
                     for code in target_procedures
                     if self.procedure_code_samples[code] is not None]
        else:
            lines = np.random.choice(self.size, size=1).tolist()

        # ✅ Nếu rỗng (code hiếm), fallback ngẫu nhiên
        if len(lines) == 0:
            lines = np.random.choice(self.size, size=1).tolist()

        lines = np.array(lines, dtype=np.int64)

        diagnoses_data = self.diagnoses_data[lines]
        procedures_data = self.procedures_data[lines]
        lens = self.lens[lines]

        diagnoses_data = torch.from_numpy(diagnoses_data).to(self.device, dtype=torch.float)
        procedures_data = torch.from_numpy(procedures_data).to(self.device, dtype=torch.float)
        lens = torch.from_numpy(lens).to(self.device, torch.long)

        return diagnoses_data, procedures_data, lens


def get_train_sampler(train_loader, device):
    data = train_loader.dataset.data
    diagnoses_data, procedures_data, lens = data[0], data[1], data[2]
    return DataSampler(diagnoses_data, procedures_data, lens, device)
