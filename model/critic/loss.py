import torch
from torch import nn, autograd

from model.utils import sequence_mask


class WGANGPLoss(nn.Module):
    def __init__(self, discriminator, lambda_=10):
        super().__init__()
        self.discriminator = discriminator
        self.lambda_ = lambda_

    def forward(self, real_diagnoses, real_procedures, real_diag_hiddens, real_proc_hiddens, 
                fake_diagnoses, fake_procedures, fake_diag_hiddens, fake_proc_hiddens, lens):
        # Forward real data
        d_real, d_real_diag, d_real_proc = self.discriminator(
            real_diagnoses, real_procedures, real_diag_hiddens, real_proc_hiddens, lens
        )
        
        # Forward fake data
        d_fake, d_fake_diag, d_fake_proc = self.discriminator(
            fake_diagnoses, fake_procedures, fake_diag_hiddens, fake_proc_hiddens, lens
        )
        
        # Calculate gradient penalty
        gradient_penalty = self.get_gradient_penalty(
            real_diagnoses, real_procedures, real_diag_hiddens, real_proc_hiddens,
            fake_diagnoses, fake_procedures, fake_diag_hiddens, fake_proc_hiddens, lens
        )
        
        # Calculate Wasserstein distance
        wasserstein_distance = d_real.mean() - d_fake.mean()
        d_loss = -wasserstein_distance + gradient_penalty
        
        return d_loss, wasserstein_distance, d_real_diag, d_real_proc, d_fake_diag, d_fake_proc

    def get_gradient_penalty(self, real_diagnoses, real_procedures, real_diag_hiddens, real_proc_hiddens,
                           fake_diagnoses, fake_procedures, fake_diag_hiddens, fake_proc_hiddens, lens):
        batch_size = len(real_diagnoses)
        
        with torch.no_grad():
            alpha = torch.rand((batch_size, 1, 1)).to(real_diagnoses.device)
            
            # Interpolate diagnoses
            interpolates_diagnoses = alpha * real_diagnoses + (1 - alpha) * fake_diagnoses
            interpolates_diag_hiddens = alpha * real_diag_hiddens + (1 - alpha) * fake_diag_hiddens
            
            # Interpolate procedures
            interpolates_procedures = alpha * real_procedures + (1 - alpha) * fake_procedures
            interpolates_proc_hiddens = alpha * real_proc_hiddens + (1 - alpha) * fake_proc_hiddens
        
        # Set requires_grad for gradient computation
        interpolates_diagnoses = autograd.Variable(interpolates_diagnoses, requires_grad=True)
        interpolates_procedures = autograd.Variable(interpolates_procedures, requires_grad=True)
        interpolates_diag_hiddens = autograd.Variable(interpolates_diag_hiddens, requires_grad=True)
        interpolates_proc_hiddens = autograd.Variable(interpolates_proc_hiddens, requires_grad=True)
        
        # Forward through discriminator
        disc_interpolates, _, _ = self.discriminator(
            interpolates_diagnoses, interpolates_procedures, 
            interpolates_diag_hiddens, interpolates_proc_hiddens, lens
        )
        
        # Compute gradients
        gradients = autograd.grad(
            outputs=disc_interpolates, 
            inputs=[interpolates_diagnoses, interpolates_procedures, interpolates_diag_hiddens, interpolates_proc_hiddens],
            grad_outputs=torch.ones_like(disc_interpolates),
            create_graph=True, 
            retain_graph=True
        )
        
        # Concatenate and compute gradient penalty
        gradients = torch.cat([g.reshape(len(g), -1) for g in gradients], dim=-1)
        gradient_penalty = (gradients.norm(2, dim=1) - 1) ** 2
        gradient_penalty = gradient_penalty.mean() * self.lambda_
        
        return gradient_penalty

    # Thêm method để tương thích với code cũ
    def forward_combined(self, real_data, real_hiddens, fake_data, fake_hiddens, lens):
        """Method tương thích - split combined input"""
        # Giả sử real_data, fake_data là combined: [diagnoses, procedures]
        diagnosis_num = real_data.shape[2] // 2
        
        real_diagnoses = real_data[:, :, :diagnosis_num]
        real_procedures = real_data[:, :, diagnosis_num:]
        fake_diagnoses = fake_data[:, :, :diagnosis_num]
        fake_procedures = fake_data[:, :, diagnosis_num:]
        
        # Giả sử hiddens cũng cần chia tương tự
        real_diag_hiddens = real_hiddens[:, :, :real_hiddens.shape[2]//2]
        real_proc_hiddens = real_hiddens[:, :, real_hiddens.shape[2]//2:]
        fake_diag_hiddens = fake_hiddens[:, :, :fake_hiddens.shape[2]//2]
        fake_proc_hiddens = fake_hiddens[:, :, fake_hiddens.shape[2]//2:]
        
        return self.forward(
            real_diagnoses, real_procedures, real_diag_hiddens, real_proc_hiddens,
            fake_diagnoses, fake_procedures, fake_diag_hiddens, fake_proc_hiddens, lens
        )