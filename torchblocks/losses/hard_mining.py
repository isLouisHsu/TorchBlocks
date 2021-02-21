import torch
import torch.nn as nn


class HardMining(nn.Module):
    def __init__(self, loss_fct=nn.CrossEntropyLoss(reduction='none'), save_rate=2, reduction='mean'):
        super(HardMining, self).__init__()
        self.save_rate = save_rate
        self.loss_fct = loss_fct
        self.reduction = reduction

    def forward(self, logits, target):
        batch_size = logits.shape[0]
        loss = self.loss_fct(logits, target)
        ind_sorted = torch.argsort(-loss)  # from big to small
        num_saved = int(self.save_rate * batch_size)
        ind_update = ind_sorted[:num_saved]
        loss = loss.index_select(0, ind_update)

        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
