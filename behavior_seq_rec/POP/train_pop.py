import sys, os
import re
import numpy as np
import utils
import argparse
from POP import POP
import glob
import pandas as pd



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', help='The directory of input', type=str, default='../data/')
    # parser.add_argument('--output_dir', help='The directory of output', type=str, default='model/pop_model/')
    parser.add_argument('--model_name', help='Model name ', type=str, default='pop_model')
    parser.add_argument('--target_behavior', help='Target behavior', type=str, default='buy')
    args = parser.parse_args()

    data_dir = args.data_dir
    # o_dir = args.output_dir
    model_name = args.model_name
    target_behavior = args.target_behavior

    # csv_file = glob.glob(data_dir + '/*.csv')[0]
    # csv_file = data_dir+'/data.csv'
    data_file = glob.glob(data_dir + '/data.txt')[0]
    # u_behavior_df = pd.read_csv(csv_file)
    # groupby_date_df = utils.preprocess_df(u_behavior_df, target_behavior)
    # train_instances = utils.create_lines_from_df(groupby_date_df)
    train_instances = utils.read_instances_lines_from_file(data_file)
    print("---------------------@Build knowledge-------------------------------")
    MAX_SEQ_LENGTH, item_dict, reversed_item_dict, item_probs, item_freq_dict, user_dict = utils.build_knowledge(train_instances)
    pop_model = POP(item_dict, reversed_item_dict, item_probs)
    o_dir = 'model/pop_model/'
    if not os.path.exists(o_dir):
        os.makedirs(o_dir)
    if target_behavior == 'buy':
        model_name = 'buy_model'
    # model for recommend cart target
    elif target_behavior == 'cart':
        model_name = 'cart_model'
    # model for recommend pv target
    elif target_behavior == 'pv':
        model_name = 'pv_model'
    else:
        print("Don't support this behavior")
        sys.exit(0)

    save_path = o_dir+'/'+model_name+'.pkl'
    utils.save_pop_model(pop_model, save_path)
    print("Save model in ", save_path)