import sys, os
import re
import numpy as np
import utils
import POP
import argparse
from POP import POP

def build_knowledge(training_instances):
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


def POP_hit_ratio(test_instances, topk, pop_model):
    list_predict_item = pop_model.top_popular_item(topk)
    hit_count = 0
    # user_dict = dict()
    # user_correct = set()
    for line in test_instances:
        elements = line.split("|")
        # user = elements[0]
        # if user not in user_dict:
        #     user_dict[user] = len(user_dict)
        basket_seq = elements[1:]
        last_basket = basket_seq[-1]
        item_list = re.split('[\\s]+', last_basket.strip())
        num_correct = len(set(item_list).intersection(list_predict_item))
        if num_correct > 0:
            hit_count += 1
            # user_correct.add(user)
    return hit_count / len(test_instances)


def POP_recall(test_instances, topk, pop_model):
    list_predict_item = pop_model.top_popular_item(topk)
    # total_correct = 0
    # total_user_correct = 0
    list_recall = []
    for line in test_instances:
        elements = line.split("|")
        user = elements[0]
        basket_seq = elements[1:]
        last_basket = basket_seq[-1]
        item_list = re.split('[\\s]+', last_basket.strip())
        num_correct = len(set(item_list).intersection(list_predict_item))
        # total_correct += num_correct
        # if num_correct > 0:
        #   total_user_correct += 1
        list_recall.append(num_correct / len(item_list))
    return np.array(list_recall).mean()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', help='The directory of input', type=str, default='../data/')
    parser.add_argument('--output_dir', help='The directory of output', type=str, default='../saved_models/')
    parser.add_argument('--model_name', help='Model name ', type=str, default='pop_model')
    args = parser.parse_args()

    data_dir = args.input_dir
    o_dir = args.output_dir
    model_name = args.model_name

    train_data_path = data_dir+'train_lines.txt'
    train_instances = utils.read_instances_lines_from_file(train_data_path)
    nb_train = len(train_instances)
    print(nb_train)

    test_data_path = data_dir+'test_lines.txt'
    test_instances = utils.read_instances_lines_from_file(test_data_path)
    nb_test = len(test_instances)
    print(nb_test)

    ### build knowledge ###
    # common_instances = train_instances + test_instances
    print("---------------------@Build knowledge-------------------------------")
    MAX_SEQ_LENGTH, item_dict, reversed_item_dict, item_probs, item_freq_dict, user_dict = build_knowledge(train_instances)
    pop_model = POP(item_dict, reversed_item_dict, item_probs)
    if not os.path.exists(o_dir):
        os.makedirs(o_dir)
    pop_model.save(o_dir)
    # saved_file = os.path.join(o_dir, model_name)
    print("Save model in ", o_dir)