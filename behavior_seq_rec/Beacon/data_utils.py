import torch
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader, Dataset, TensorDataset
import numpy as np
import re

####################  Utils for pre-process data   #######################

def create_binary_vector(item_list, item_dict):
    v = np.zeros(len(item_dict), dtype='int32')
    for item in item_list:
        v[item_dict[item]] = 1
    return v


def get_epoch(x):
    idx = x.index('_') + 1
    return int(x[idx:])


def compute_total_batches(nb_intances, batch_size):
    total_batches = int(nb_intances / batch_size)
    if nb_intances % batch_size != 0:
        total_batches += 1
    return total_batches


def seq_generator(raw_lines, item_dict):
    O = []
    S = []
    L = []
    Y = []

    lines = raw_lines[:]

    for line in lines:
        elements = line.split("|")

        # label = float(elements[0])
        bseq = elements[1:-1]
        tbasket = elements[-1]

        # Keep the length for dynamic_rnn
        L.append(len(bseq))

        # Keep the original last basket
        O.append(tbasket)

        # Add the target basket
        target_item_list = re.split('[\\s]+', tbasket.strip())
        Y.append(create_binary_vector(target_item_list, item_dict))

        s = []
        for basket in bseq:
            item_list = re.split('[\\s]+', basket.strip())
            id_list = [item_dict[item] for item in item_list]
            s.append(id_list)
        S.append(s)

    return {'S': np.asarray(S), 'L': np.asarray(L), 'Y': np.asarray(Y), 'O': np.asarray(O)}


def get_sparse_tensor_info(x, is_bseq=False):
    indices = []
    if is_bseq:
        for sid, bseq in enumerate(x):
            for t, basket in enumerate(bseq):
                for item_id in basket:
                    indices.append([sid, t, item_id])
    else:
        for bid, basket in enumerate(x):
            for item_id in basket:
                indices.append([bid, item_id])

    values = torch.ones(len(indices), dtype=torch.float32)
    indices = torch.IntTensor(indices)

    return indices, values

def generate_data_loader(data_instances, b_size, item_dict, max_seq_len, is_bseq, is_shuffle):
    data_seq = seq_generator(data_instances, item_dict)
    sparse_seq_indices, sparse_seq_values = get_sparse_tensor_info(data_seq['S'], is_bseq)
    sparse_seq_indices = torch.transpose(sparse_seq_indices, 0, 1)

    sparse_X = torch.sparse_coo_tensor(indices = sparse_seq_indices, values = sparse_seq_values,
                                       size=[len(data_seq['S']), max_seq_len, len(item_dict)])
    print(sparse_X)

    Y = torch.FloatTensor(data_seq['Y'])
    print(Y.shape)

    seq_len = torch.IntTensor(data_seq['L'])

    dataset = TensorDataset(sparse_X, seq_len, Y)
    data_loader = DataLoader(dataset=dataset, batch_size= b_size, shuffle= is_shuffle)
    return data_loader

def create_lines_from_df(groupby_date_df):
    user_behavior_dict = dict()
    for i in range(len(groupby_date_df)):
        uid = groupby_date_df.loc[i, 'user_id']
        basket = groupby_date_df.loc[i, 'list ib']
        if uid not in user_behavior_dict:
            user_behavior_dict[uid] = []
            user_behavior_dict[uid].append(basket)
        else:
            user_behavior_dict[uid].append(basket)
    lines = []
    for u in user_behavior_dict:
        line = str(u)
        b_seq = user_behavior_dict[u]
        for basket in b_seq:
            line += ' |'
            for ib_pair in basket:
                line += (' ' + str(ib_pair[0]) + ':' + str(ib_pair[1]))
        lines.append(line)
    return lines

def data_dl_filter_target_behavior(lines, behavior, filter_string):
    filtered_lines = []
    for line in lines:
        elements = line.split("|")
        user = elements[0]
        basket_seq = elements[1:]
        if len(basket_seq) < 2:
            continue
        list_basket_contain_behavior = []
        for i in range(0, len(basket_seq)):
            cur_basket = basket_seq[i]
            if behavior not in cur_basket:
                continue
            list_basket_contain_behavior.append(basket_seq[i])
        filtered_line = user
        if len(list_basket_contain_behavior) < 2:
            continue
        cur_basket = list_basket_contain_behavior[-1]
        prev_baskets = list_basket_contain_behavior[:-1]
#         if i+1 < mc_order:
#             prev_baskets = basket_seq[i - st:i]
#         else:
#             prev_baskets = basket_seq[i-mc_order:i]

        for i in range(0, len(prev_baskets)):
            prev_baskets[i] = re.sub(filter_string, '', prev_baskets[i])
            if len(prev_baskets[i].strip()) > 1:
                filtered_line += ("|"+prev_baskets[i])

#             filtered_line += "|".join(prev_baskets)
        filtered_line += "|"
        cur_item_list = [p for p in re.split('[\\s]+', cur_basket.strip())]
        for item in cur_item_list:
            if behavior in item:
                filtered_line += (' '+item)
        if len(filtered_line.split("|")) > 2:
            filtered_lines.append(filtered_line)
    return filtered_lines

def clean_lines(lines, replace_term):
    new_lines = []
    for line in lines:
        new_lines.append(re.sub(replace_term, '', line))
    return new_lines

def preprocess_df(u_behavior_df):
    new_behavior_df = u_behavior_df.sort_values(['user_id', 'date'])
    new_behavior_df['pair_ib'] = new_behavior_df[['item_id', 'behavior']].apply(tuple, axis=1)
    new_behavior_df.drop(['item_id', 'behavior'], axis=1, inplace=True)
    groupby_date_df = new_behavior_df.groupby(['user_id', 'date'])['pair_ib'].apply(list).reset_index(name='list ib')
    return groupby_date_df

def create_new_line(line):
    new_lines = []
    elements = line.split("|")
#     print(elements)
    if len(elements) == 3:
        new_lines.append(line)
        return new_lines
    else:
        for i in range(2, len(elements)):
            new_seq = "|".join(elements[:i+1])
            new_lines.append(new_seq)
    return new_lines

def create_new_data(lines):
    new_lines = []
    for line in lines:
        new_lines += create_new_line(line)
    return new_lines