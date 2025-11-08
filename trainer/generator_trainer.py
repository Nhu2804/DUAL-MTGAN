import torch


class GeneratorTrainer:
    def __init__(self, generator, critic, batch_size, train_num, lr, betas, decay_step, decay_rate):
        self.generator = generator
        self.critic = critic
        self.batch_size = batch_size
        self.train_num = train_num

        self.diagnosis_num = self.generator.diagnosis_num
        self.procedure_num = self.generator.procedure_num
        self.total_codes = self.diagnosis_num + self.procedure_num
        
        self.optimizer = torch.optim.Adam(generator.parameters(), lr=lr, betas=betas)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=decay_step, gamma=decay_rate)
        self.device = self.generator.device

    def _step(self, target_diagnoses, target_procedures, lens):
        noise = self.generator.get_noise(len(lens))
        
        # Forward qua dual-stream generator
        (diagnosis_samples, procedure_samples), (diagnosis_hiddens, procedure_hiddens) = self.generator(
            target_diagnoses, target_procedures, lens, noise
        )
        
        # Forward qua dual-stream critic
        output, diagnosis_output, procedure_output = self.critic(
            diagnosis_samples, procedure_samples, diagnosis_hiddens, procedure_hiddens, lens
        )
        
        loss = -output.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss, diagnosis_output, procedure_output

    def step(self, target_diagnoses, target_procedures, lens):
        self.generator.train()
        self.critic.eval()

        total_loss = 0
        total_diagnosis_output = 0
        total_procedure_output = 0
        
        for _ in range(self.train_num):
            loss, diagnosis_output, procedure_output = self._step(target_diagnoses, target_procedures, lens)
            total_loss += loss.item()
            total_diagnosis_output += diagnosis_output.mean().item()
            total_procedure_output += procedure_output.mean().item()
            
        total_loss /= self.train_num
        total_diagnosis_output /= self.train_num
        total_procedure_output /= self.train_num
        
        self.scheduler.step()

        return total_loss, total_diagnosis_output, total_procedure_output

    # Thêm method để tương thích với code cũ
    def step_combined(self, target_codes, lens):
        """Method tương thích - giả định target_codes là combined"""
        # Cần logic để chia target_codes thành diagnoses và procedures
        target_diagnoses = target_codes  # Tạm thời
        target_procedures = target_codes  # Tạm thời
        return self.step(target_diagnoses, target_procedures, lens)