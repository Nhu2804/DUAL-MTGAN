import torch

from model.base_model import BaseModel
from model.utils import sequence_mask
from .generator_layers import GRU, SmoothCondition


class Generator(BaseModel):
    def __init__(self, diagnosis_num, procedure_num, hidden_dim, attention_dim, max_len, device=None):
        super().__init__(param_file_name='generator.pt')
        self.diagnosis_num = diagnosis_num
        self.procedure_num = procedure_num
        self.total_codes = diagnosis_num + procedure_num
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim
        self.max_len = max_len
        self.device = device

        self.noise_dim = hidden_dim
        
        # Dual-stream GRU
        self.gru = GRU(diagnosis_num, procedure_num, hidden_dim, max_len, device)
        
        # Dual-stream smooth condition
        self.smooth_condition = SmoothCondition(diagnosis_num, procedure_num, attention_dim)

    def forward(self, target_diagnoses, target_procedures, lens, noise):
        # Forward qua dual GRU
        (diagnosis_samples, procedure_samples), (diagnosis_hiddens, procedure_hiddens) = self.gru(noise)
        
        # Apply smooth condition
        diagnosis_samples, procedure_samples = self.smooth_condition(
            diagnosis_samples, procedure_samples, lens, target_diagnoses, target_procedures
        )
        
        samples = (diagnosis_samples, procedure_samples)
        hiddens = (diagnosis_hiddens, procedure_hiddens)
        
        return samples, hiddens

    def sample(self, target_diagnoses, target_procedures, lens, noise=None, return_hiddens=False):
        if noise is None:
            noise = self.get_noise(len(lens))
        with torch.no_grad():
            mask = sequence_mask(lens, self.max_len).unsqueeze(dim=-1)
            (diagnosis_prob, procedure_prob), (diagnosis_hiddens, procedure_hiddens) = self.forward(
                target_diagnoses, target_procedures, lens, noise
            )
            
            # Sample diagnoses và procedures
            diagnosis_samples = torch.bernoulli(diagnosis_prob).to(diagnosis_prob.dtype)
            procedure_samples = torch.bernoulli(procedure_prob).to(procedure_prob.dtype)
            
            # Apply mask
            diagnosis_samples *= mask
            procedure_samples *= mask
            
            samples = (diagnosis_samples, procedure_samples)
            
            if return_hiddens:
                diagnosis_hiddens *= mask
                procedure_hiddens *= mask
                hiddens = (diagnosis_hiddens, procedure_hiddens)
                return samples, hiddens
            else:
                return samples

    def get_noise(self, batch_size):
        noise = torch.randn(batch_size, self.noise_dim).to(self.device)
        return noise

    def get_target_codes(self, batch_size, code_type='both'):
        """Lấy target codes cho diagnoses, procedures, hoặc cả hai"""
        if code_type == 'diagnosis' or code_type == 'both':
            target_diagnoses = torch.randint(low=0, high=self.diagnosis_num, size=(batch_size,))
        else:
            target_diagnoses = None
            
        if code_type == 'procedure' or code_type == 'both':
            target_procedures = torch.randint(low=0, high=self.procedure_num, size=(batch_size,))
        else:
            target_procedures = None
            
        return target_diagnoses, target_procedures

    # Thêm method để tương thích với code cũ
    def forward_combined(self, target_codes, lens, noise):
        """Method tương thích - giả định target_codes là combined"""
        # Chia target_codes thành diagnoses và procedures (cần logic phù hợp)
        target_diagnoses = target_codes  # Tạm thời, cần logic chia phù hợp
        target_procedures = target_codes  # Tạm thời, cần logic chia phù hợp
        return self.forward(target_diagnoses, target_procedures, lens, noise)