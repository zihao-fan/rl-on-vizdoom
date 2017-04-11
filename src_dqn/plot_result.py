#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
import utils
import json
import os
import numpy as np
import matplotlib.pyplot as plt
result_dir = '../basic_result'
ls_list = ['g^', 'bs', 'r-.', 'c+', 'kx']

def load_list_from_json(filename):
    with open(os.path.join(result_dir, filename), 'r') as f:
        res = json.load(f)
        return np.asarray(res)

def plot_mean_std(res_dict):
    my_models = res_dict.keys()
    index = 0
    for model in my_models:
        plt.errorbar(x=res_dict[model][:, 0],
            y=res_dict[model][:, 1],
            yerr=res_dict[model][:, 2],
            fmt=ls_list[index])
        index += 1
    plt.legend(my_models, loc='lower right')
    plt.title('simple_basic.cfg')
    plt.xlabel('Time / mins')
    plt.ylabel('Reward mean/std')
    plt.show()

def plot_min(res_dict):
    my_models = res_dict.keys()
    index = 0
    for model in my_models:
        plt.plot(res_dict[model][:, 0],
            res_dict[model][:,3], ls_list[index])
        index+=1
    plt.legend(my_models, loc='lower right')
    plt.title('simple_basic.cfg')
    plt.xlabel('Time / mins')
    plt.ylabel('Reward min')
    plt.show()

if __name__ == '__main__':
    res_dict = {}
    result_jsons = utils.get_filenames_from_dir(result_dir)
    titles = ['stage2_2', 'stage4', 'stage2_1', 'stage3', 'stage2_2_scratch']
    pair_list = zip(titles, result_jsons)
    for pair in pair_list:
        res_dict[pair[0]] = load_list_from_json(pair[1])
    plot_mean_std(res_dict)
    plot_min(res_dict)