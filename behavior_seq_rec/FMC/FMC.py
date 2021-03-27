import pickle
import numpy as np
class FMC():
    def __init__(self, item_dict, reversed_item_dict, item_freq_dict, n_factor, mc_order, W, H):
        self.item_freq_dict = item_freq_dict
        self.item_dict = item_dict
        self.reversed_item_dict = reversed_item_dict

        self.item_set = set(item_dict.values())
        self.n_item = len(item_dict)
        self.mc_order = mc_order

        self.n_factor = n_factor
        self.W = W
        self.H = H

    @staticmethod
    def dump(fpmcObj, fname):
        pickle.dump(fpmcObj, open(fname, 'wb'))

    @staticmethod
    def load(fname):
        return pickle.load(open(fname, 'rb'))

    def save(self, filename):
        '''Save the parameters of a network into a file
        '''
        print('Save model in ' + filename)
        # if not os.path.exists(os.path.dirname(filename)):
        #   os.makedirs(os.path.dirname(filename))
        np.savez(filename, W=self.W, H=self.H)

    def load(self, filename):
        '''Load parameters values form a file
        '''
        f = np.load(filename)
        self.W = f['W']
        print(self.W.shape)
        self.H = f['H']
        print(self.H.shape)

    def top_predicted_item(self, previous_basket, topk):
        # candidate = np.zeros(self.n_item)
        prev_basket_idx = [self.item_dict[item] for item in previous_basket]
        # for item_idx in prev_basket_idx:
        # for i in range(self.n_item):
        candidate = np.matmul(self.W[prev_basket_idx,:], self.H).mean(axis=0)
        topk_idx = np.argpartition(candidate, -topk)[-topk:]
        sorted_topk_idx = topk_idx[np.argsort(candidate[topk_idx])]
        topk_item = [self.reversed_item_dict[item] for item in sorted_topk_idx]
        # print("Done")
        return topk_item