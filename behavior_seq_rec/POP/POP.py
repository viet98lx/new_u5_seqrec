import numpy as np
import json
import os

class POP:
  def __init__(self, item_dict, reversed_item_dict, item_probs):
    self.list_pop = item_probs
    self.item_dict = item_dict
    self.reversed_item_dict = reversed_item_dict

  def top_popular_item(self, topk):
    topk_idx = np.argpartition(self.list_pop, -topk)[-topk:]
    topk_item = [self.reversed_item_dict[item] for item in topk_idx]

    return topk_item
