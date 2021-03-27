from FMC import FMC
import argparse
import scipy.sparse as sp
import os
import FMC_utils
import json
import sys
import glob
import pandas as pd
from sklearn.decomposition import non_negative_factorization

def train_FMC_model(train_instances, n_factor, model_name, o_dir):
    w_behavior = {'buy': 1, 'cart': 1, 'fav': 1, 'pv': 1}
    print("---------------------@Build knowledge-------------------------------")
    MAX_SEQ_LENGTH, item_dict, reversed_item_dict, item_probs, item_freq_dict, user_dict = FMC_utils.build_knowledge(train_instances)
    print('Build knowledge done')
    mc_order = 1
    max_iter = 500
    transition_matrix = FMC_utils.calculate_transition_matrix(train_instances, item_dict, item_freq_dict, reversed_item_dict, mc_order)
    W, H, n_iter = non_negative_factorization(transition_matrix, n_components=n_factor, init='random', random_state=0,
                                              solver='mu', beta_loss='kullback-leibler', max_iter=max_iter)
    if not os.path.exists(o_dir):
        os.makedirs(o_dir)

    fmc_model = FMC(item_dict, reversed_item_dict, item_freq_dict, n_factor, mc_order, W, H)
    save_path = o_dir+'/'+model_name+'.pkl'
    FMC_utils.save_model(fmc_model, save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', help='The directory of input', type=str, default='../data/')
    # parser.add_argument('--output_dir', help='The directory of output', type=str, default='model/fmc_model/')
    parser.add_argument('--n_factor', help='Factor of decomposition', type=int, default=16)
    parser.add_argument('--target_behavior', help='Target behavior', type=str, default='buy')
    # parser.add_argument('--type_model', help='Target behavior', type=str, default='mc')

    args = parser.parse_args()
    data_dir = args.data_dir
    n_factor = args.n_factor
    # o_dir = args.output_dir
    target_behavior = args.target_behavior
    # csv_file = glob.glob(data_dir + '/*.csv')[0]
    # csv_file = data_dir + '/data.csv'
    data_file = glob.glob(data_dir + '/data.txt')[0]
    # u_behavior_df = pd.read_csv(csv_file)
    # groupby_date_df = FMC_utils.preprocess_df(u_behavior_df)
    # raw_lines = FMC_utils.create_lines_from_df(groupby_date_df)
    raw_lines = FMC_utils.read_instances_lines_from_file(data_file)
    mc_order = 1

    # model for recommend buy target
    if target_behavior == 'buy':
        filter_string = '[\\s][0-9]+:fav|[\\s][0-9]+:pv'
        filtered_lines = FMC_utils.filter_target_behavior(raw_lines, mc_order, target_behavior, filter_string)
        o_dir = 'model/fmc_model/'
        model_name = 'buy_model'
        # model_name = 'buy_seqrec'
        train_FMC_model(filtered_lines, n_factor, model_name, o_dir)
    # model for recommend cart target
    elif target_behavior == 'cart':
        filter_string = '[\\s][0-9]+:buy'
        filtered_lines = FMC_utils.filter_target_behavior(raw_lines, mc_order, target_behavior, filter_string)
        o_dir = 'model/fmc_model/'
        model_name = 'cart_model'
        # model_name = 'cart_seqrec'
        train_FMC_model(filtered_lines, n_factor, model_name, o_dir)
    # model for recommend pv target
    elif target_behavior == 'pv':
        filter_string = '[\\s][0-9]+:buy|[\\s][0-9]+:fav|[\\s][0-9]+:cart'
        filtered_lines = FMC_utils.filter_target_behavior(raw_lines, mc_order, target_behavior, filter_string)
        o_dir = 'model/fmc_model/'
        model_name = 'pv_model'
        # model_name = 'pv_seqrec'
        train_FMC_model(filtered_lines, n_factor, model_name, o_dir)
    else:
        print("Don't support this behavior")
        sys.exit(0)

