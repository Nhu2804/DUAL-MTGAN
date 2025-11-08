import torch

from model import WGANGPLoss


class CriticTrainer:
    def __init__(self, critic, generator, base_gru, batch_size, train_num, lr, lambda_, betas, decay_step, decay_rate):
        self.critic = critic
        self.generator = generator
        self.base_gru = base_gru
        self.batch_size = batch_size
        self.train_num = train_num

        self.optimizer = torch.optim.Adam(critic.parameters(), lr=lr, betas=betas)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=decay_step, gamma=decay_rate)
        self.loss_fn = WGANGPLoss(critic, lambda_=lambda_)

    def _step(self, real_diagnoses, real_procedures, real_lens, target_diagnoses, target_procedures):
        # Tính real hidden states từ base GRU
        real_diag_hiddens, real_proc_hiddens = self.base_gru.calculate_hidden(
            real_diagnoses, real_procedures, real_lens
        )
        
        # Generate fake data từ generator
        (fake_diagnoses, fake_procedures), (fake_diag_hiddens, fake_proc_hiddens) = self.generator.sample(
            target_diagnoses, target_procedures, real_lens, return_hiddens=True
        )
        
        # Tính loss với dual-stream data
        loss, wasserstein_distance, d_real_diag, d_real_proc, d_fake_diag, d_fake_proc = self.loss_fn(
            real_diagnoses, real_procedures, real_diag_hiddens, real_proc_hiddens,
            fake_diagnoses, fake_procedures, fake_diag_hiddens, fake_proc_hiddens, real_lens
        )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return (loss.item(), wasserstein_distance.item(), 
                d_real_diag.mean().item(), d_real_proc.mean().item(),
                d_fake_diag.mean().item(), d_fake_proc.mean().item())

    def step(self, real_diagnoses, real_procedures, real_lens, target_diagnoses, target_procedures):
        self.critic.train()
        self.generator.eval()

        total_loss, total_w_distance = 0, 0
        total_d_real_diag, total_d_real_proc = 0, 0
        total_d_fake_diag, total_d_fake_proc = 0, 0
        
        for _ in range(self.train_num):
            (loss_i, w_distance_i, 
             d_real_diag_i, d_real_proc_i,
             d_fake_diag_i, d_fake_proc_i) = self._step(
                real_diagnoses, real_procedures, real_lens, target_diagnoses, target_procedures
            )
            
            total_loss += loss_i
            total_w_distance += w_distance_i
            total_d_real_diag += d_real_diag_i
            total_d_real_proc += d_real_proc_i
            total_d_fake_diag += d_fake_diag_i
            total_d_fake_proc += d_fake_proc_i
            
        # Tính average
        avg_loss = total_loss / self.train_num
        avg_w_distance = total_w_distance / self.train_num
        avg_d_real_diag = total_d_real_diag / self.train_num
        avg_d_real_proc = total_d_real_proc / self.train_num
        avg_d_fake_diag = total_d_fake_diag / self.train_num
        avg_d_fake_proc = total_d_fake_proc / self.train_num
        
        self.scheduler.step()

        return (avg_loss, avg_w_distance, 
                avg_d_real_diag, avg_d_real_proc,
                avg_d_fake_diag, avg_d_fake_proc)

    def evaluate(self, data_loader, device):
        self.critic.train()
        with torch.no_grad():
            total_score = 0
            total_diagnosis_score = 0
            total_procedure_score = 0
            
            for data in data_loader:
                # Data structure mới: (diagnoses_data, procedures_data, lens)
                diagnoses_data, procedures_data, lens = data
                diagnoses_data = diagnoses_data.to(device)
                procedures_data = procedures_data.to(device)
                lens = lens.to(device)
                
                # Tính hidden states
                diag_hiddens, proc_hiddens = self.base_gru.calculate_hidden(
                    diagnoses_data, procedures_data, lens
                )
                
                # Evaluate critic
                score, diagnosis_score, procedure_score = self.critic(
                    diagnoses_data, procedures_data, diag_hiddens, proc_hiddens, lens
                )
                
                total_score += score.mean().item()
                total_diagnosis_score += diagnosis_score.mean().item()
                total_procedure_score += procedure_score.mean().item()
                
            avg_score = -total_score / len(data_loader)
            avg_diagnosis_score = -total_diagnosis_score / len(data_loader)
            avg_procedure_score = -total_procedure_score / len(data_loader)
            
            return avg_score, avg_diagnosis_score, avg_procedure_score

    # Thêm method để tương thích với code cũ
    def step_combined(self, real_data, real_lens, target_codes):
        """Method tương thích - split combined input"""
        # Cần logic chia real_data và target_codes thành diagnoses/procedures
        diagnosis_num = real_data.shape[2] // 2
        
        real_diagnoses = real_data[:, :, :diagnosis_num]
        real_procedures = real_data[:, :, diagnosis_num:]
        target_diagnoses = target_codes  # Tạm thời
        target_procedures = target_codes  # Tạm thời
        
        return self.step(real_diagnoses, real_procedures, real_lens, target_diagnoses, target_procedures)