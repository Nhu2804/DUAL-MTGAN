import torch

from model import PredictNextLoss


class BaseGRUTrainer:
    def __init__(self, args, base_gru, max_len, train_loader, params_path):
        self.base_gru = base_gru
        self.train_loader = train_loader
        self.params_path = params_path

        self.epochs = args.base_gru_epochs

        self.optimizer = torch.optim.Adam(base_gru.parameters(), lr=args.base_gru_lr)
        self.loss_fn = PredictNextLoss(max_len)

    def train(self):
        print('pre-training base gru...')
        for epoch in range(1, self.epochs + 1):
            print('Epoch %d / %d:' % (epoch, self.epochs))
            total_loss = 0.0
            total_diagnosis_loss = 0.0
            total_procedure_loss = 0.0
            total_num = 0
            steps = len(self.train_loader)
            
            for step, data in enumerate(self.train_loader, start=1):
                # Data structure mới từ DatasetRealNext: 
                # (diag_x, diag_lens, diag_y, proc_x, proc_lens, proc_y)
                diag_x, diag_lens, diag_y, proc_x, proc_lens, proc_y = data
                
                # Forward qua dual-stream BaseGRU
                diagnosis_pred, procedure_pred = self.base_gru(diag_x, proc_x)
                
                # Tính loss cho cả hai streams
                total_loss, diagnosis_loss, procedure_loss = self.loss_fn(
                    diagnosis_pred, procedure_pred, diag_y, proc_y, diag_lens
                )

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                total_loss += total_loss.item() * len(diag_x)
                total_diagnosis_loss += diagnosis_loss.item() * len(diag_x)
                total_procedure_loss += procedure_loss.item() * len(diag_x)
                total_num += len(diag_x)

                print('\r    Step %d / %d, total_loss: %.4f, diag_loss: %.4f, proc_loss: %.4f' % 
                      (step, steps, total_loss / total_num, total_diagnosis_loss / total_num, total_procedure_loss / total_num), 
                      end='')

            print('\r    Step %d / %d, total_loss: %.4f, diag_loss: %.4f, proc_loss: %.4f' % 
                  (steps, steps, total_loss / total_num, total_diagnosis_loss / total_num, total_procedure_loss / total_num))
        
        self.base_gru.save(self.params_path)