import sys, os, pickle, argparse
import re
import random
import FMC_utils
from FMC import FMC
import glob
import pandas as pd

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', help='The directory of input', type=str, default='../data/')
    parser.add_argument('--output_dir', help='The directory of output', type=str, default='../saved_models/')
    parser.add_argument('--model_name', help='Model name ', type=str, default='fpmc')
    parser.add_argument('--nb_predict', help='# of predict', type=int, default=50)
    parser.add_argument('--topk', help='# of predict', type=int, default=10)
    parser.add_argument('--mc_order', help='Markov order', type=int, default=1)
    parser.add_argument('--target_behavior', help='Target behavior', type=str, default='buy')
    args = parser.parse_args()
    # parser.add_argument('--example_file', help='Example_file', type=str, default=None)
    args = parser.parse_args()

    data_dir = args.data_dir
    o_dir = args.output_dir
    model_name = args.model_name
    nb_predict = args.nb_predict
    topk = args.topk
    mc_order = args.mc_order
    target_behavior = args.target_behavior

    save_path = o_dir + '/' + model_name + '.pkl'
    fmc_model = FMC_utils.load_model(save_path)

    csv_file = glob.glob(data_dir + '/*test.csv')
    u_behavior_df = pd.read_csv(csv_file)
    groupby_date_df = FMC_utils.preprocess_df(u_behavior_df)
    raw_lines = FMC_utils.create_lines_from_df(groupby_date_df)

    if target_behavior == 'buy':
        filter_string = '[\\s][0-9]+:fav|[\\s][0-9]+:buy'
        filtered_lines = FMC_utils.filter_target_behavior(raw_lines, mc_order, target_behavior, filter_string)
    # model for recommend cart target
    elif target_behavior == 'cart':
        filter_string = '[\\s][0-9]+:buy'
        filtered_lines = FMC_utils.filter_target_behavior(raw_lines, mc_order, target_behavior, filter_string)
    # model for recommend pv target
    elif target_behavior == 'pv':
        filter_string = '[\\s][0-9]+:buy|[\\s][0-9]+:fav|[\\s][0-9]+:cart'
        filtered_lines = FMC_utils.filter_target_behavior(raw_lines, mc_order, target_behavior, filter_string)

    topk = 50
    print('Predict to outfile')
    predict_file = os.path.join(o_dir, 'predict_' + model_name + '.txt')
    FMC_utils.write_predict(predict_file, filtered_lines, topk, fmc_model)
    print('Predict done')
    ground_truth, predict = FMC_utils.read_predict(predict_file)
    for topk in [5, 10, 15, 20]:
        print("Top : ", topk)
        # hit_rate = MC_hit_ratio(test_instances, topk, mc_model)
        # recall = MC_recall(test_instances, topk, mc_model)
        hit_rate = FMC_utils.hit_ratio(ground_truth, predict, topk)
        recall = FMC_utils.recall(ground_truth, predict, topk)
        print("hit ratio: ", hit_rate)
        print("recall: ", recall)

