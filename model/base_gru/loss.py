from torch import nn

from model.utils import sequence_mask


class PredictNextLoss(nn.Module):
    def __init__(self, max_len):
        super().__init__()
        self.max_len = max_len
        self.loss_fn = nn.BCELoss(reduction='none')

    def forward(self, diagnosis_pred, procedure_pred, diagnosis_label, procedure_label, lens):
        mask = sequence_mask(lens, self.max_len).unsqueeze(dim=-1)
        
        # Calculate loss cho diagnoses
        diagnosis_loss = self.loss_fn(diagnosis_pred, diagnosis_label)
        diagnosis_loss = diagnosis_loss * mask
        diagnosis_loss = diagnosis_loss.sum(dim=-1).sum(dim=-1).mean()
        
        # Calculate loss cho procedures
        procedure_loss = self.loss_fn(procedure_pred, procedure_label)
        procedure_loss = procedure_loss * mask
        procedure_loss = procedure_loss.sum(dim=-1).sum(dim=-1).mean()
        
        # Total loss (có thể điều chỉnh weights nếu cần)
        total_loss = diagnosis_loss + procedure_loss
        
        return total_loss, diagnosis_loss, procedure_loss

    # Thêm method để tương thích với code cũ (nếu cần)
    def forward_combined(self, pred, label, lens):
        """Method để tương thích với code cũ - split combined input"""
        # Giả sử pred và label là combined: [diagnoses, procedures]
        diagnosis_num = pred.shape[2] // 2  # Giả định diagnoses và procedures bằng nhau hoặc biết trước
        
        diagnosis_pred = pred[:, :, :diagnosis_num]
        procedure_pred = pred[:, :, diagnosis_num:]
        diagnosis_label = label[:, :, :diagnosis_num]
        procedure_label = label[:, :, diagnosis_num:]
        
        return self.forward(diagnosis_pred, procedure_pred, diagnosis_label, procedure_label, lens)