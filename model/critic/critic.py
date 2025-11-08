import torch
from torch import nn

from model.base_model import BaseModel
from model.utils import sequence_mask


class Critic(BaseModel):
    def __init__(self, diagnosis_num, procedure_num, hidden_dim, generator_hidden_dim, max_len):
        super().__init__(param_file_name='critic.pt')

        self.diagnosis_num = diagnosis_num
        self.procedure_num = procedure_num
        self.total_codes = diagnosis_num + procedure_num
        self.hidden_dim = hidden_dim
        self.generator_hidden_dim = generator_hidden_dim
        self.max_len = max_len

        # Dual linear networks: một cho diagnoses, một cho procedures
        self.diagnosis_linear = nn.Sequential(
            nn.Linear(diagnosis_num + generator_hidden_dim, hidden_dim),
            nn.ELU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )
        
        self.procedure_linear = nn.Sequential(
            nn.Linear(procedure_num + generator_hidden_dim, hidden_dim),
            nn.ELU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )
        
        # Combiner để kết hợp scores từ cả hai streams
        self.combiner = nn.Linear(2, 1)

    def forward(self, diagnoses_x, procedures_x, diagnosis_hiddens, procedure_hiddens, lens):
        # Process diagnoses stream
        diagnosis_input = torch.cat([diagnoses_x, diagnosis_hiddens], dim=-1)
        diagnosis_output = self.diagnosis_linear(diagnosis_input).squeeze(dim=-1)
        
        # Process procedures stream
        procedure_input = torch.cat([procedures_x, procedure_hiddens], dim=-1)
        procedure_output = self.procedure_linear(procedure_input).squeeze(dim=-1)
        
        # Apply mask và tính average
        mask = sequence_mask(lens, self.max_len)
        
        diagnosis_output = diagnosis_output * mask
        diagnosis_output = diagnosis_output.sum(dim=-1) / lens
        
        procedure_output = procedure_output * mask
        procedure_output = procedure_output.sum(dim=-1) / lens
        
        # Kết hợp scores từ cả hai streams
        combined_scores = torch.stack([diagnosis_output, procedure_output], dim=-1)
        final_output = self.combiner(combined_scores).squeeze(dim=-1)
        
        return final_output, diagnosis_output, procedure_output

    # Thêm method để tương thích với code cũ
    def forward_combined(self, x, hiddens, lens):
        """Method tương thích - split combined input"""
        # Giả sử x là combined: [diagnoses, procedures]
        diagnosis_x = x[:, :, :self.diagnosis_num]
        procedure_x = x[:, :, self.diagnosis_num:]
        
        # Giả sử hiddens là combined hoặc cần logic chia phù hợp
        diagnosis_hiddens = hiddens[:, :, :self.generator_hidden_dim]
        procedure_hiddens = hiddens[:, :, :self.generator_hidden_dim]  # Hoặc logic chia khác
        
        return self.forward(diagnosis_x, procedure_x, diagnosis_hiddens, procedure_hiddens, lens)
