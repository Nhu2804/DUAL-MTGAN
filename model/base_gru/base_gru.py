import torch
from torch import nn

from model.base_model import BaseModel
from model.utils import sequence_mask


class BaseGRU(BaseModel):
    def __init__(self, diagnosis_num, procedure_num, hidden_dim, max_len):
        super().__init__(param_file_name='base_gru.pt')
        
        # Dual GRUs: một cho diagnoses, một cho procedures
        self.diagnosis_gru = nn.GRU(input_size=diagnosis_num, hidden_size=hidden_dim, batch_first=True)
        self.procedure_gru = nn.GRU(input_size=procedure_num, hidden_size=hidden_dim, batch_first=True)
        
        # Dual linear layers
        self.diagnosis_linear = nn.Sequential(
            nn.Linear(hidden_dim, diagnosis_num),
            nn.Sigmoid()
        )
        self.procedure_linear = nn.Sequential(
            nn.Linear(hidden_dim, procedure_num),
            nn.Sigmoid()
        )
        
        self.max_len = max_len
        self.diagnosis_num = diagnosis_num
        self.procedure_num = procedure_num

    def forward(self, diagnoses_x, procedures_x):
        # Forward diagnoses stream
        diagnosis_outputs, _ = self.diagnosis_gru(diagnoses_x)
        diagnosis_pred = self.diagnosis_linear(diagnosis_outputs)
        
        # Forward procedures stream  
        procedure_outputs, _ = self.procedure_gru(procedures_x)
        procedure_pred = self.procedure_linear(procedure_outputs)
        
        return diagnosis_pred, procedure_pred

    def calculate_hidden(self, diagnoses_x, procedures_x, lens):
        with torch.no_grad():
            # Lấy kích thước thực tế
            batch_size = diagnoses_x.size(0)
            seq_len = diagnoses_x.size(1)
            
            
            # Tạo mask với seq_len thực tế
            mask = sequence_mask(lens, seq_len)
            
            mask = mask.unsqueeze(dim=-1)
            
            # Calculate hidden states
            diagnosis_outputs, _ = self.diagnosis_gru(diagnoses_x)
            procedure_outputs, _ = self.procedure_gru(procedures_x)
            
            # Apply mask
            diagnosis_hidden = diagnosis_outputs * mask
            procedure_hidden = procedure_outputs * mask
            
            return diagnosis_hidden, procedure_hidden

    # Thêm method để tương thích với code cũ (nếu cần)
    def forward_combined(self, x):
        """Method để tương thích với code cũ - split combined input"""
        # Giả sử x là combined: [diagnoses, procedures]
        diagnoses_x = x[:, :, :self.diagnosis_num]
        procedures_x = x[:, :, self.diagnosis_num:]
        return self.forward(diagnoses_x, procedures_x)