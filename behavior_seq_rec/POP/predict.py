import sys, os, pickle, argparse
import re
import random
import utils
import numpy as np
from POP import POP
import glob
import pandas as pd

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', help='The directory of input', type=str, default='../data/')
    parser.add_argument('--model_dir', help='The directory of output', type=str, default='../saved_models/')
    parser.add_argument('--model_name', help='Model name ', type=str, default='')
    parser.add_argument('--nb_predict', help='# of predict', type=int, default=10)
    parser.add_argument('--target_behavior', help='Target behavior', type=str, default='buy')
    parser.add_argument('--topk', help='# of predict', type=int, default=10)
    args = parser.parse_args()

    data_dir = args.data_dir
    o_dir = args.model_dir
    model_name = args.model_name
    nb_predict = args.nb_predict
    topk = args.topk

    save_path = o_dir+'/'+model_name+'.pkl'
    pop_model = utils.load_pop_model(save_path)

    csv_file = glob.glob(data_dir + '/*test.csv')
    u_behavior_df = pd.read_csv(csv_file)
    groupby_date_df = utils.preprocess_df(u_behavior_df)
    test_instances = utils.create_lines_from_df(groupby_date_df)

    topk = 50
    print('Predict to outfile')
    predict_file = os.path.join(o_dir, 'predict_' + model_name + '.txt')
    utils.write_predict(predict_file, test_instances, topk, pop_model)
    print('Predict done')
    ground_truth, predict = utils.read_predict(predict_file)
    for topk in [5, 10, 15, 20]:
        print("Top : ", topk)
        # hit_rate = MC_hit_ratio(test_instances, topk, mc_model)
        # recall = MC_recall(test_instances, topk, mc_model)
        hit_rate = utils.hit_ratio(ground_truth, predict, topk)
        recall = utils.recall(ground_truth, predict, topk)
        print("hit ratio: ", hit_rate)
        print("recall: ", recall)