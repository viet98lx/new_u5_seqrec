import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

###################  Multi Intent Next Basket Rec Model ####################

class RecSysModel(torch.nn.Module):
    """
      Input data: [u_1, u_2, u_3, u_4, u_5, u_i..., u_n]
      u_i: ith user's seq
      u_i = [b_1, b_2, b_3, b_4, b_j...b_n]
      b_j: jth user's basket which contains items: i_1, i_2, i_3, ....i_n.
    """

    def __init__(self, config_param, max_seq_length, item_dict, reversed_item_dict, device, d_type):
        super(RecSysModel, self).__init__()
        self.embedding_dim = config_param['embedding_dim']
        self.max_seq_length = max_seq_length
        self.nb_items = len(item_dict)
        self.item_dict = item_dict
        self.reversed_item_dict = reversed_item_dict
        self.batch_size = config_param['batch_size']
        self.top_k = config_param['top_k']
        self.dropout = config_param['dropout']
        self.device = device
        self.d_type = d_type

        # network architecture
        self.drop_out_1 = nn.Dropout(p=self.dropout)
        self.fc_basket_encoder_1 = nn.Linear(in_features=self.nb_items, out_features=self.embedding_dim, bias=True)
        self.h2item_score = nn.Linear(in_features=self.embedding_dim, out_features=self.nb_items, bias=True)
        self.init_weight()

    def init_weight(self):
        torch.nn.init.kaiming_uniform_(self.fc_basket_encoder_1.weight.data, nonlinearity='relu')
        self.fc_basket_encoder_1.bias.data.zero_()

        torch.nn.init.xavier_uniform_(self.h2item_score.weight.data)
        self.h2item_score.bias.data.zero_()

    def forward(self, x):
        batch_size = x.size()[0]

        basket_encoder_1 = F.relu(self.fc_basket_encoder_1(x))

        hidden_to_score = self.h2item_score(basket_encoder_1)

        # predict next items score
        next_item_probs = torch.sigmoid(hidden_to_score)
        return next_item_probs
