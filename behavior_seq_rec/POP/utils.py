import scipy.sparse as sp
import numpy as np
import torch
import os, re
import pickle
import sys, os
sys.path.append(os.path.abspath(os.path.join('..', 'data')))


################## utils and build knowledge about data ###################

def build_knowledge(training_instances):
    MAX_SEQ_LENGTH = 0
    item_freq_dict = {}
    user_dict = dict()

    for line in training_instances:
        elements = line.split("|")

        if len(elements) - 1 > MAX_SEQ_LENGTH:
            MAX_SEQ_LENGTH = len(elements) - 1

        user = elements[0]
        if user not in user_dict:
            user_dict[user] = len(user_dict)

        basket_seq = elements[1:]

        for basket in basket_seq:
            item_list = re.split('[\\s]+', basket.strip())
            for item_obs in item_list:
                if item_obs not in item_freq_dict:
                    item_freq_dict[item_obs] = 1
                else:
                    item_freq_dict[item_obs] += 1

    items = sorted(list(item_freq_dict.keys()))
    item_dict = dict()
    item_probs = []
    for item in items:
        item_dict[item] = len(item_dict)
        item_probs.append(item_freq_dict[item])

    item_probs = np.asarray(item_probs, dtype=np.float32)
    item_probs /= np.sum(item_probs)

    reversed_item_dict = dict(zip(item_dict.values(), item_dict.keys()))
    return MAX_SEQ_LENGTH, item_dict, reversed_item_dict, item_probs, item_freq_dict, user_dict

def add_tuple(t, pairs):
    assert len(t) == 2
    if t[0] != t[1]:
        if t not in pairs:
            pairs[t] = 1
        else:
            pairs[t] += 1

# def create_sparse_matrix(pairs, NB_ITEMS):
#     row = [p[0] for p in pairs]
#     col = [p[1] for p in pairs]
#     data = [pairs[p] for p in pairs]
#     adj_matrix = sp.csc_matrix((data, (row, col)), shape=(NB_ITEMS, NB_ITEMS), dtype="float32")
#     nb_nonzero = len(pairs)
#     density = nb_nonzero * 1.0 / NB_ITEMS / NB_ITEMS
#     print("Density of first order matrix: {:.6f}".format(density))
#
#     return sp.csc_matrix(adj_matrix, dtype="float32")
#
# def create_identity_matrix(nb_items):
#     return sp.identity(nb_items, dtype="float32").tocsr()
#
# def create_zero_matrix(nb_items):
#     return sp.csr_matrix((nb_items, nb_items), dtype="float32")
#
# def normalize_adj(adj_matrix):
#     """Symmetrically normalize adjacency matrix."""
#     row_sum = np.array(adj_matrix.sum(1))
#     d_inv_sqrt = np.power(row_sum, -0.5).flatten()
#     d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
#     d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
#     normalized_matrix = adj_matrix.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
#     return normalized_matrix.tocsr()
#
# def remove_diag(adj_matrix):
#     new_adj_matrix = sp.csr_matrix(adj_matrix)
#     new_adj_matrix.setdiag(0.0)
#     new_adj_matrix.eliminate_zeros()
#     return new_adj_matrix

def read_instances_lines_from_file(file_path):
    with open(file_path, "r") as f:
        lines = [line.rstrip('\n') for line in f]
        return lines

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
                # print(ib_pair)
                line += (' ' + str(ib_pair))
        lines.append(line)
    return lines

def preprocess_df(u_behavior_df, target_behavior):
    new_behavior_df = u_behavior_df.sort_values(['user_id', 'date'])
    # new_behavior_df['pair_ib'] = new_behavior_df[['item_id', 'behavior']].apply(tuple, axis=1)
    # new_behavior_df.drop(['item_id', 'behavior'], axis=1, inplace=True)
    new_behavior_df = new_behavior_df[new_behavior_df['behavior'] == target_behavior]
    groupby_date_df = new_behavior_df.groupby(['user_id', 'date'])['item_id'].apply(list).reset_index(name='list ib')
    return groupby_date_df

def save_pop_model(pop_model, save_path):
    with open(save_path, 'wb') as output:
        pickle.dump(pop_model, output, 3)

def write_predict(file_name, test_instances, topk, model):
    f = open(file_name, 'w')
    for line in test_instances:
        elements = line.split("|")
        user = elements[0]
        basket_seq = elements[1:-1]
        last_basket = basket_seq[-1]
        # prev_basket = basket_seq[-2]
        prev_item_list = []
        for basket in basket_seq:
            prev_item_list.append([p.split(':')[0] for p in re.split('[\\s]+', basket.strip())])
        list_predict_item = model.top_popular_item(topk)
        # item_list = re.split('[\\s]+', last_basket.strip())
        cur_item_list = [p.split(':')[0] for p in re.split('[\\s]+', last_basket.strip())]
        f.write(str(user)+'\n')
        f.write('ground_truth:')
        for item in cur_item_list:
            f.write(' '+str(item))
        f.write('\n')
        f.write('predicted:')
        predict_len = len(list_predict_item)
        for i in range(predict_len):
            f.write(' '+str(list_predict_item[predict_len-1-i]))
        f.write('\n')
    f.close()

def read_predict(file_name):
    f = open(file_name, 'r')
    lines = f.readlines()
    list_ground_truth_basket = []
    list_predict_basket = []
    for i in range(0, len(lines), 3):
        user = lines[i].strip('\n')
        list_ground_truth_basket.append(re.split('[\\s]+',lines[i+1].strip('\n'))[1:])
        list_predict_basket.append(re.split('[\\s]+',lines[i+2].strip('\n'))[1:])

    return list_ground_truth_basket, list_predict_basket

def hit_ratio(list_ground_truth_basket, list_predict_basket, topk):
    hit_count = 0
    for gt, predict in zip(list_ground_truth_basket, list_predict_basket):
        num_correct = len(set(gt).intersection(predict[:topk]))
        if num_correct > 0:
            hit_count += 1
            # user_correct.add(user)
    return hit_count / len(list_ground_truth_basket)

def recall(list_ground_truth_basket, list_predict_basket, topk):
    list_recall = []
    for gt, predict in zip(list_ground_truth_basket, list_predict_basket):
        num_correct = len(set(gt).intersection(predict[:topk]))
        list_recall.append(num_correct / len(gt))
    return np.array(list_recall).mean()

def load_pop_model(save_path):
    with open(save_path, 'rb') as input:
        restored_model = pickle.load(input)
    return restored_model