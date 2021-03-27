import itertools
import scipy.sparse as sp
import re
import numpy as np
import time
import pickle

def calculate_transition_matrix(train_instances, item_dict, item_freq_dict, reversed_item_dict, w_behavior, mc_order):
  pair_dict = dict()
  NB_ITEMS = len(item_dict)
  print("number items: ", NB_ITEMS)
  list_pair_dict = [dict() for _ in range(mc_order)]
  start = time.time()
  for line in train_instances:
      elements = line.split("|")
      user = elements[0]
      # print('User')
      basket_seq = elements[1:]
      cur_basket = basket_seq[-1]
      cur_item_list = [p.split(':')[0] for p in re.split('[\\s]+', cur_basket.strip())]
      cur_item_idx = [item_dict[item] for item in cur_item_list]
      prev_baskets = basket_seq[:-1]
      # prev_item_list = []
      nb_prev_baskets = len(prev_baskets)
      for i in range(len(prev_baskets)-1,-1,-1):
        prev_item_list = [(p.split(':')) for p in re.split('[\\s]+', prev_baskets[i].strip())]
        prev_ib_idx = [(item_dict[ib[0]], ib[1]) for ib in prev_item_list]
        for t in list(itertools.product(prev_ib_idx, cur_item_idx)):
            item_pair = (t[0][0], t[1])
            if item_pair in list_pair_dict[nb_prev_baskets-1-i].keys():
                list_pair_dict[nb_prev_baskets-1-i][item_pair] += w_behavior[t[0][1]]
            else:
                list_pair_dict[nb_prev_baskets-1-i][item_pair] = w_behavior[t[0][1]]
  end = time.time()
  print("Time to run all seq line: ", end-start)

  start_1 = time.time()
  list_trans_matrix = []
  for pair_dict in list_pair_dict:
      for key in pair_dict.keys():
        pair_dict[key] /= item_freq_dict[reversed_item_dict[key[0]]]

      row = [p[0] for p in pair_dict]
      col = [p[1] for p in pair_dict]
      data = [pair_dict[p] for p in pair_dict]
      transition_matrix = sp.csr_matrix((data, (row, col)), shape=(NB_ITEMS, NB_ITEMS), dtype="float32")
      nb_nonzero = len(pair_dict)
      density = nb_nonzero * 1.0 / NB_ITEMS / NB_ITEMS
      print("Density of matrix: {:.6f}".format(density))
      list_trans_matrix.append(transition_matrix)
  end_1 = time.time()
  print("Time to Create transition matrix: ", end_1-start_1)

  return list_trans_matrix

def build_knowledge(training_instances, w_behavior):
    MAX_SEQ_LENGTH = 0
    item_freq_dict = {}
    user_dict = dict()

    for line in training_instances:
        elements = line.split("|")

        if len(elements) - 1 > MAX_SEQ_LENGTH:
            MAX_SEQ_LENGTH = len(elements) - 1

        user = elements[0]
        user_dict[user] = len(user_dict)

        basket_seq = elements[1:]

        for basket in basket_seq:
            ib_pair = [tuple(p.split(':')) for p in re.split('[\\s]+', basket.strip())]
            # print(ib_pair)
            for item_obs in ib_pair:
                if item_obs[0] not in item_freq_dict:
                    # print(item_obs[0])
                    # print(item_obs[1])
                    item_freq_dict[item_obs[0]] = w_behavior[item_obs[1]]
                else:
                    item_freq_dict[item_obs[0]] += w_behavior[item_obs[1]]

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

def save_model(model, save_path):
    with open(save_path, 'wb') as output:
        pickle.dump(model, output, 3)

def load_model(save_path):
    with open(save_path, 'rb') as input:
        restored_model = pickle.load(input)
    return restored_model

def write_predict(file_name, test_instances, topk, MC_model):
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
        list_predict_item = MC_model.top_predicted_mc_order(prev_item_list, topk)
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
                line += (' ' + str(ib_pair[0]) + ':' + str(ib_pair[1]))
        lines.append(line)
    return lines


def filter_target_behavior(lines, mc_order, target_behavior, filter_string):
    filtered_lines = []
    for line in lines:
        elements = line.split("|")
        user = elements[0]
        basket_seq = elements[1:]
        if len(basket_seq) < 2:
            continue
        st = 1
        for i in range(st, len(basket_seq)):
            filtered_line = user
            cur_basket = basket_seq[i]
            if target_behavior not in cur_basket:
                continue
            if i + 1 < mc_order:
                prev_baskets = basket_seq[i - st:i]
            else:
                prev_baskets = basket_seq[i - mc_order:i]

            for i in range(0, len(prev_baskets)):
                prev_baskets[i] = re.sub(filter_string, '', prev_baskets[i])
                if len(prev_baskets[i].strip()) > 1:
                    filtered_line += ("|" + prev_baskets[i])

            #             filtered_line += "|".join(prev_baskets)
            filtered_line += "|"
            cur_item_list = [p for p in re.split('[\\s]+', cur_basket.strip())]
            for item in cur_item_list:
                if target_behavior in item:
                    filtered_line += (' ' + item)
            if len(filtered_line.split("|")) > 2:
                filtered_lines.append(filtered_line)
    return filtered_lines

def preprocess_df(u_behavior_df):
    new_behavior_df = u_behavior_df.sort_values(['user_id', 'date'])
    new_behavior_df['pair_ib'] = new_behavior_df[['item_id', 'behavior']].apply(tuple, axis=1)
    new_behavior_df.drop(['item_id', 'behavior'], axis=1, inplace=True)
    groupby_date_df = new_behavior_df.groupby(['user_id', 'date'])['pair_ib'].apply(list).reset_index(name='list ib')
    return groupby_date_df