import torch
from torch import nn

from model.utils import MaskedAttention


class GRU(nn.Module):
    def __init__(self, diagnosis_num, procedure_num, hidden_dim, max_len, device=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_len = max_len
        self.device = device
        self.diagnosis_num = diagnosis_num
        self.procedure_num = procedure_num

        # Dual GRU cells: một cho diagnoses, một cho procedures
        self.diagnosis_gru_cell = nn.GRUCell(input_size=diagnosis_num, hidden_size=hidden_dim)
        self.procedure_gru_cell = nn.GRUCell(input_size=procedure_num, hidden_size=hidden_dim)
        
        # Dual output layers
        self.diagnosis_hidden2codes = nn.Sequential(
            nn.Linear(hidden_dim, diagnosis_num),
            nn.Sigmoid()
        )
        self.procedure_hidden2codes = nn.Sequential(
            nn.Linear(hidden_dim, procedure_num),
            nn.Sigmoid()
        )

    def step(self, diagnosis_x, procedure_x, h=None):
        # Step cho diagnoses
        diagnosis_h_n = self.diagnosis_gru_cell(diagnosis_x, h)
        diagnosis_codes = self.diagnosis_hidden2codes(diagnosis_h_n)
        
        # Step cho procedures
        procedure_h_n = self.procedure_gru_cell(procedure_x, h)
        procedure_codes = self.procedure_hidden2codes(procedure_h_n)
        
        return diagnosis_codes, procedure_codes, diagnosis_h_n, procedure_h_n

    def forward(self, noise):
        # Khởi tạo từ noise
        diagnosis_codes = self.diagnosis_hidden2codes(noise)
        procedure_codes = self.procedure_hidden2codes(noise)
        
        h = torch.zeros(len(noise), self.hidden_dim, device=self.device)
        
        diagnosis_samples, procedure_samples = [], []
        diagnosis_hiddens, procedure_hiddens = [], []
        
        for _ in range(self.max_len):
            diagnosis_samples.append(diagnosis_codes)
            procedure_samples.append(procedure_codes)
            
            diagnosis_codes, procedure_codes, diagnosis_h, procedure_h = self.step(
                diagnosis_codes, procedure_codes, h
            )
            
            diagnosis_hiddens.append(diagnosis_h)
            procedure_hiddens.append(procedure_h)
        
        # Stack samples và hidden states
        diagnosis_samples = torch.stack(diagnosis_samples, dim=1)
        procedure_samples = torch.stack(procedure_samples, dim=1)
        diagnosis_hiddens = torch.stack(diagnosis_hiddens, dim=1)
        procedure_hiddens = torch.stack(procedure_hiddens, dim=1)

        return (diagnosis_samples, procedure_samples), (diagnosis_hiddens, procedure_hiddens)


class SmoothCondition(nn.Module):
    def __init__(self, diagnosis_num, procedure_num, attention_dim):
        super().__init__()
        self.diagnosis_num = diagnosis_num
        self.procedure_num = procedure_num
        
        # Dual attention mechanisms
        self.diagnosis_attention = MaskedAttention(diagnosis_num, attention_dim)
        self.procedure_attention = MaskedAttention(procedure_num, attention_dim)

    def forward(self, diagnosis_x, procedure_x, lens, target_diagnoses=None, target_procedures=None):
        # Apply attention cho diagnoses nếu có target
        if target_diagnoses is not None:
            diagnosis_score = self.diagnosis_attention(diagnosis_x, lens)
            diagnosis_score_tensor = torch.zeros_like(diagnosis_x)
            diagnosis_score_tensor[torch.arange(len(diagnosis_x)), :, target_diagnoses] = diagnosis_score
            diagnosis_x = diagnosis_x + diagnosis_score_tensor
            diagnosis_x = torch.clip(diagnosis_x, max=1)
        
        # Apply attention cho procedures nếu có target
        if target_procedures is not None:
            procedure_score = self.procedure_attention(procedure_x, lens)
            procedure_score_tensor = torch.zeros_like(procedure_x)
            procedure_score_tensor[torch.arange(len(procedure_x)), :, target_procedures] = procedure_score
            procedure_x = procedure_x + procedure_score_tensor
            procedure_x = torch.clip(procedure_x, max=1)
        
        return diagnosis_x, procedure_x