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

        diag_num = self.generator.diagnosis_num
        proc_num = self.generator.procedure_num
        hidden_dim = self.generator.hidden_dim

        # =====================================================
        # 🧩 Diagnosis stream
        # =====================================================
        (diagnosis_samples, _), (diagnosis_hiddens, _) = self.generator(
            target_diagnoses, target_procedures * 0, lens, noise
        )

        # 🧠 zeros_proc phải đúng số chiều của procedure stream
        zeros_proc = torch.zeros(diagnosis_samples.size(0), diagnosis_samples.size(1), proc_num, device=self.device)
        zeros_proc_h = torch.zeros(diagnosis_hiddens.size(0), diagnosis_hiddens.size(1), hidden_dim, device=self.device)

        output_diag, diagnosis_output, _ = self.critic(
            diagnosis_samples, zeros_proc, diagnosis_hiddens, zeros_proc_h, lens
        )
        g_loss_diag = -output_diag.mean()

        # =====================================================
        # 🧩 Procedure stream
        # =====================================================
        (_, procedure_samples), (_, procedure_hiddens) = self.generator(
            target_diagnoses * 0, target_procedures, lens, noise
        )

        zeros_diag = torch.zeros(procedure_samples.size(0), procedure_samples.size(1), diag_num, device=self.device)
        zeros_diag_h = torch.zeros(procedure_hiddens.size(0), procedure_hiddens.size(1), hidden_dim, device=self.device)

        output_proc, _, procedure_output = self.critic(
            zeros_diag, procedure_samples, zeros_diag_h, procedure_hiddens, lens
        )
        g_loss_proc = -output_proc.mean()

        # =====================================================
        # 🧠 Combine hai nhánh (trung bình)
        # =====================================================
        g_loss = (g_loss_diag + g_loss_proc) / 2

        self.optimizer.zero_grad()
        g_loss.backward()
        self.optimizer.step()

        return g_loss, diagnosis_output, procedure_output



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