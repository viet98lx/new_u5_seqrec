import torch
################## Loss Funtion and compute recall fucntion #####################

class Weighted_BCE_Loss(torch.nn.Module):

    def __init__(self, reduction ='mean'):
        super(Weighted_BCE_Loss, self).__init__()
        self.reduction = reduction

    def forward(self, sigmoid_predict, y, pos_weight=None):
        neg_y = (1.0 - y)
        nb_pos = (neg_y == 0).sum(dim=-1).to(sigmoid_predict.dtype)
        nb_neg = (y.size()[1] - nb_pos).to(sigmoid_predict.dtype)
        ratio = (nb_neg / nb_pos).unsqueeze(dim=-1)
        # print(ratio)

        neg_loss = -neg_y * torch.log(1.0 - sigmoid_predict)
        pos_loss = -y * torch.log(sigmoid_predict)
        loss_batch = pos_loss * ratio + neg_loss + 1e-8

        if (self.reduction == 'mean'):
            return loss_batch.mean()
        else:
            return loss_batch.sum()
