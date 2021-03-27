from FMC import FMC
import argparse
import scipy.sparse as sp
import os
import FMC_utils
import json
from sklearn.decomposition import non_negative_factorization

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', help='The directory of input', type=str, default='../data/')
    parser.add_argument('--output_dir', help='The directory of output', type=str, default='../saved_models/')
    parser.add_argument('--model_name', help='Model name ', type=str, default='fpmc')
    parser.add_argument('--n_factor', help='# of factor', type=int, default=4)
    parser.add_argument('--w_behavior', help='Weight behavior file', type=str, default=None)
    parser.add_argument('--mc_order', help='Markov order', type=int, default=1)
    parser.add_argument('--max_iter', help='max iter of factorization', type=int, default=300)
    parser.add_argument('--transition_matrix_path', help='The directory of transition matrix', type=str, default=None)
    args = parser.parse_args()

    data_dir = args.input_dir
    o_dir = args.output_dir
    model_name = args.model_name
    transition_matrix_path = args.transition_matrix_path
    n_factor = args.n_factor
    mc_order = args.mc_order
    max_iter = args.max_iter
    w_behavior_file = args.w_behavior

    if w_behavior_file is None:
        w_behavior = {'buy': 1, 'cart': 1, 'fav': 1, 'pv':1}
    else:
        with open(w_behavior_file, 'r') as fp:
            w_behavior = json.load(fp)

    train_data_path = data_dir+'train_lines.txt'
    train_instances = FMC_utils.read_instances_lines_from_file(train_data_path)
    nb_train = len(train_instances)
    print(nb_train)

    test_data_path = data_dir+'test_lines.txt'
    test_instances = FMC_utils.read_instances_lines_from_file(test_data_path)
    nb_test = len(test_instances)
    print(nb_test)

    ### build knowledge ###
    # common_instances = train_instances + test_instances
    print("---------------------@Build knowledge-------------------------------")
    MAX_SEQ_LENGTH, item_dict, reversed_item_dict, item_probs, item_freq_dict, user_dict = FMC_utils.build_knowledge(train_instances+test_instances)
    if transition_matrix_path is None:
        transition_matrix = FMC_utils.calculate_transition_matrix(train_instances, item_dict, item_freq_dict, reversed_item_dict)
        sp_matrix_path = model_name+'_transition_matrix_MC.npz'
        if not os.path.exists(o_dir):
            os.makedirs(o_dir)
        saved_file = os.path.join(o_dir, sp_matrix_path)
        print("Save model in ", saved_file)
        sp.save_npz(saved_file, transition_matrix)
    else:
        transition_matrix = sp.load_npz(transition_matrix_path)
    if not os.path.exists(o_dir):
        os.makedirs(o_dir)
    W, H, n_iter = non_negative_factorization(transition_matrix, n_components=n_factor, init='random', random_state=0,
                                              solver='mu', beta_loss='kullback-leibler', max_iter=max_iter)
    # np.savez(o_dir+'W_matrix_64_factor.npz', W_matrix=W)
    # np.savez(o_dir+'H_matrix_64_factor.npz', H_matrix=H)
    print("n_iter: ", n_iter)
    fmc_model = FMC(item_dict, reversed_item_dict, item_freq_dict, n_factor, mc_order)
    fmc_model.W = W
    fmc_model.H = H
    fmc_model.save(os.path.join(o_dir,model_name+'_W_H'))
    topk = 50
    print('Predict to outfile')
    predict_file = os.path.join(o_dir, 'predict_' + model_name + '.txt')
    FMC_utils.write_predict(predict_file, test_instances, topk, fmc_model)
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