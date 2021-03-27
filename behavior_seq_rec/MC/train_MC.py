import os, glob
import pandas as pd
import re
import sys
import numpy as np
import MC_utils
from MC import MarkovChain
import argparse
import scipy.sparse as sp
import os
import json

def train_MC_model(train_instances, model_name, o_dir):
    w_behavior = {'buy': 1, 'cart': 1, 'fav': 1, 'pv': 1}
    print("---------------------@Build knowledge-------------------------------")
    MAX_SEQ_LENGTH, item_dict, reversed_item_dict, item_probs, item_freq_dict, user_dict = MC_utils.build_knowledge(
        train_instances, w_behavior)
    print('Build knowledge done')
    mc_order = 1
    list_transition_matrix = MC_utils.calculate_transition_matrix(train_instances, item_dict, item_freq_dict,
                                                                  reversed_item_dict, w_behavior, mc_order)

    if not os.path.exists(o_dir):
        os.makedirs(o_dir)

    mc_model = MarkovChain(item_dict, reversed_item_dict, item_freq_dict, w_behavior, list_transition_matrix, mc_order)
    save_path = o_dir+'/'+model_name+'.pkl'
    MC_utils.save_model(mc_model, save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', help='The directory of input', type=str, default='../data/')
    # parser.add_argument('--output_dir', help='The directory of output', type=str, default='model/mc_model/')
    parser.add_argument('--target_behavior', help='Target behavior', type=str, default='buy')
    # parser.add_argument('--type_model', help='Target behavior', type=str, default='mc')

    args = parser.parse_args()
    data_dir = args.data_dir
    # o_dir = args.output_dir
    target_behavior = args.target_behavior
    # csv_file = glob.glob(data_dir + '/*.csv')[0]
    data_file = glob.glob(data_dir + '/data.txt')[0]
    # csv_file = data_dir + '/data.csv'
    # u_behavior_df = pd.read_csv(data_file)
    # groupby_date_df = MC_utils.preprocess_df(u_behavior_df)
    # raw_lines = MC_utils.create_lines_from_df(groupby_date_df)
    raw_lines = MC_utils.read_instances_lines_from_file(data_file)
    mc_order = 1

    # model for recommend buy target
    if target_behavior == 'buy':
        filter_string = '[\\s][0-9]+:fav|[\\s][0-9]+:pv'
        filtered_lines = MC_utils.filter_target_behavior(raw_lines, mc_order, target_behavior, filter_string)
        o_dir = 'model/mc_model/'
        model_name = 'buy_model'
        train_MC_model(filtered_lines, model_name, o_dir)
    # model for recommend cart target
    elif target_behavior == 'cart':
        filter_string = '[\\s][0-9]+:buy'
        filtered_lines = MC_utils.filter_target_behavior(raw_lines, mc_order, target_behavior, filter_string)
        o_dir = 'model/mc_model/'
        model_name = 'cart_model'
        train_MC_model(filtered_lines, model_name, o_dir)
    # model for recommend pv target
    elif target_behavior == 'pv':
        filter_string = '[\\s][0-9]+:buy|[\\s][0-9]+:fav|[\\s][0-9]+:cart'
        filtered_lines = MC_utils.filter_target_behavior(raw_lines, mc_order, target_behavior, filter_string)
        o_dir = 'model/mc_model/'
        model_name = 'pv_model'
        train_MC_model(filtered_lines, model_name, o_dir)
    else:
        print("Don't support this behavior")
        sys.exit(0)

