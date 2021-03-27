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

    def __init__(self, config_param, max_seq_length, item_probs, adj_matrix, device, d_type):
        super(RecSysModel, self).__init__()
        self.rnn_units = config_param['rnn_units']
        self.rnn_layers = config_param['rnn_layers']
        self.embedding_dim = config_param['embedding_dim']
        self.max_seq_length = max_seq_length
        self.nb_items = len(item_probs)
        self.item_probs = item_probs
        self.alpha = config_param['alpha']
        self.batch_size = config_param['batch_size']
        self.top_k = config_param['top_k']
        self.dropout = config_param['dropout']
        self.device = device
        self.d_type = d_type

        # initialized adjacency matrix
        self.C = torch.from_numpy(adj_matrix)
        # values = adj_matrix.data
        # indices = np.vstack((adj_matrix.row, adj_matrix.col))
        #
        # i = torch.LongTensor(indices)
        # v = torch.FloatTensor(values)
        # shape = adj_matrix.shape
        # self.C = torch.sparse.FloatTensor(i, v, torch.Size(shape)).to(self.device)

        # threshold ignore weak correlation
        threshold = adj_matrix.mean()
        print(threshold)
        item_bias = torch.ones(self.nb_items) / self.nb_items
        print(item_bias)

        # network architecture
        self.drop_out_1 = nn.Dropout(p=self.dropout)
        self.fc_basket_encoder_1 = nn.Linear(in_features=self.nb_items, out_features=self.embedding_dim, bias=True)
        self.seq_encoder = nn.LSTM(self.embedding_dim, self.rnn_units, self.rnn_layers, bias=True, batch_first=True)
        # self.W_hidden = nn.Parameter(data=torch.randn(self.rnn_units, self.nb_items).type(self.d_type))
        self.h2item_score = nn.Linear(in_features=self.rnn_units, out_features=self.nb_items, bias=False)
        self.threshold = nn.Parameter(data=torch.Tensor([threshold]).type(d_type))
        self.I_B = nn.Parameter(data=item_bias.type(d_type))
        self.init_weight()

    def init_weight(self):
        torch.nn.init.kaiming_uniform_(self.fc_basket_encoder_1.weight.data, nonlinearity='relu')
        self.fc_basket_encoder_1.bias.data.zero_()

        torch.nn.init.xavier_uniform_(self.h2item_score.weight.data)

    def init_hidden(self, batch_size):
        # Before we've done anything, we dont have any hidden state.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        weight = next(self.parameters()).data
        return (weight.new(self.rnn_layers, batch_size, self.rnn_units).zero_(),
                weight.new(self.rnn_layers, batch_size, self.rnn_units).zero_())

    def forward(self, x, seq_len, hidden):
        batch_size = x.size()[0]
        item_bias_diag = F.relu(self.I_B)
        reshape_x = x.reshape(-1, self.nb_items)
        x_mm_C = torch.mm(reshape_x.cpu(), self.C)
        encode_x_graph = reshape_x*item_bias_diag + F.relu(x_mm_C.to(self.device) - torch.abs(self.threshold))
        basket_x = encode_x_graph.reshape(-1, self.max_seq_length, self.nb_items)
        basket_encoder_1 = self.drop_out_1(F.relu(self.fc_basket_encoder_1(basket_x)))

        # next basket sequence encoder
        lstm_out, (h_n, c_n) = self.seq_encoder(basket_encoder_1, hidden)
        actual_index = torch.arange(0, batch_size) * self.max_seq_length + (seq_len - 1)
        actual_lstm_out = lstm_out.reshape(-1, self.rnn_units)[actual_index]

        hidden_to_score = self.h2item_score(actual_lstm_out)

        # predict next items score
        next_item_probs = torch.sigmoid(hidden_to_score)

        # next_item_probs_mm_C = torch.mm(next_item_probs.cpu(), self.C)
        # + next_item_probs_mm_C.to(self.device)

        # print(next_item_probs)
        predict = (1 - self.alpha) * next_item_probs + self.alpha * (
                next_item_probs*item_bias_diag)
        return predict
