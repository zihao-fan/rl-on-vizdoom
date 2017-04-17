#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
import os
import mxnet as mx
import numpy as np
import itertools as it
from random import sample, randint, random
import skimage.color, skimage.transform
from utils import ReplayMemory, root_path

# Q-Learning Hyper-parameters
discount_factor = 0.99
replay_memory_size = 10000
resolution = (224, 224)
ctx = mx.cpu(0)
update_time = 200

# Neural network settings
batch_size = 64
model_save_dir = os.path.join(root_path, 'dqn_saved_weights')
pretrained_model_name = os.path.join(root_path, 'pretrained_models', 'Inception-BN')
frame_repeat = 12

class AgentDQN(object):
    
    def __init__(self, 
            action_count, 
            feature_extractor=None, 
            channels=1, 
            param_file=None):

        self.time_step = 0 # A internal counter for model save

        self.channels = channels
        self.feature_extractor = feature_extractor if feature_extractor != None else 'flatten_output'
        if not os.path.exists(model_save_dir):
            print('Creating directory', model_save_dir)
            os.makedirs(model_save_dir)
        self.model_savefile = os.path.join(model_save_dir, 'dqn_' + self.feature_extractor + '.weights')
        self.memory = ReplayMemory(capacity=replay_memory_size, resolution=resolution, s1_frames=channels)
        self.action_count = action_count
        self.actions = [list(a) for a in it.product([0, 1], repeat=action_count)]
        self.available_action_count = len(self.actions)
        
        # neural net
        self.pretrained_net = self.load_pretrained_network(pretrained_model_name)
        self.target = self.create_network(is_train = False)
        self.qnet = self.create_network(is_train = True)
        if param_file != None:
            self.qnet.load_params(param_file)
        self.copy_target_network()

    def load_pretrained_network(self, model_name):
        sym, arg_params, aux_params = mx.model.load_checkpoint(model_name, 126)
        return (sym, arg_params, aux_params)

    def get_fine_tune_model(self, predict=False):
        '''
            net = all_layers[$name$]
            possible_names = 'flatten_output' # stage_4
                        'ch_concat_5b_chconcat_output' 5b
                        'ch_concat_4e_chconcat_output' 4e
                        'ch_concat_3c_chconcat_output' 3c
                        'pool_2_output'
        '''
        all_layers = self.pretrained_net[0].get_internals()
        net = all_layers[self.feature_extractor]
        a = mx.sym.Variable('action')
        q_target = mx.sym.Variable('q_target')
        qvalue = mx.sym.FullyConnected(data=net,
            num_hidden=self.available_action_count, name='qvalue')

        selected_elements = qvalue * a
        sumed_value = mx.sym.sum(selected_elements, axis=1, name='value_sum')
        output = (sumed_value - q_target) ** 2
        loss = mx.sym.MakeLoss(output)

        if predict:
            return qvalue
        else:
            return loss

    def copy_target_network(self):
        arg_params, aux_params = self.qnet.get_params()
        self.target.init_params(initializer = None, arg_params= arg_params,
                                aux_params = aux_params, force_init = True)

    def get_new_args(self):
        arg_params = self.pretrained_net[1]
        aux_params = self.pretrained_net[2]
        new_args = dict({k:arg_params[k] for k in arg_params if 'qvalue' not in k})
        new_auxs = dict({k:aux_params[k] for k in aux_params if 'qvalue' not in k})
        return new_args, new_auxs

    def get_lr_dict(self, symbol):
        arg_names = symbol.list_arguments()
        lr_scale_dict = {}
        for name in arg_names:
            if name != 'data' and name != 'action' and name != 'q_target':
                lr_scale_dict[name] = 1 if name.startswith('qvalue') else 0.1
        return lr_scale_dict

    def create_network(self, bef_args = None, is_train = True):
        if is_train:
            my_symbol = self.get_fine_tune_model()
            lr_scale_dict = self.get_lr_dict(my_symbol)
            mod_q = mx.mod.Module(symbol = my_symbol, 
                                  data_names = ('data', 'action'),
                                  label_names = ('q_target',),
                                  context = ctx)
            batch = batch_size
            mod_q.bind(data_shapes = [('data', (batch, self.channels, resolution[0], resolution[1])),
                                    ('action', (batch, self.available_action_count))],
                       label_shapes = [('q_target', (batch,))],
                       for_training = is_train)
            mod_q.init_params(initializer = mx.init.Xavier(factor_type='in',magnitude=2.34), arg_params=bef_args)
            # load pretrained params
            new_args, new_auxs = self.get_new_args()
            adam = mx.optimizer.Adam()
            # for fine-tunning
            adam.set_lr_mult(lr_scale_dict)
            mod_q.set_params(new_args, new_auxs, allow_missing=True)
            mod_q.init_optimizer(optimizer = adam)
        else:
            mod_q = mx.mod.Module(symbol=self.get_fine_tune_model(predict = True),
                                  data_names = ('data',),
                                  label_names = None,
                                  context = ctx)
            batch = batch_size
            mod_q.bind(data_shapes = [('data', (batch, self.channels, resolution[0], resolution[1]))],
                     for_training = is_train)
            mod_q.init_params(initializer = mx.init.Xavier(factor_type='in', magnitude=2.34), arg_params=bef_args)
            # load pretrained params
            new_args, new_auxs = self.get_new_args()
            mod_q.set_params(new_args, new_auxs, allow_missing=True)
        return mod_q

    def learn_from_memory(self):
        if self.memory.size > batch_size:
            s1, a, s2, is_terminal, r = self.memory.get_sample(batch_size)
            self.time_step += 1

            # calculate q_target
            y_batch = np.zeros((batch_size,))
            self.target.forward(mx.io.DataBatch([mx.nd.array(s2, ctx)], []))
            q_values = self.target.get_outputs()[0].asnumpy()
            y_batch = r + discount_factor * (1 - is_terminal) * np.max(q_values, axis = 1)
            actions = np.zeros((batch_size, self.available_action_count))
            actions[np.arange(batch_size), a] = 1
            
            # train agent net
            self.qnet.forward(mx.io.DataBatch([mx.nd.array(s1, ctx), mx.nd.array(actions, ctx)],
                              [mx.nd.array(y_batch, ctx)]))
            self.qnet.backward()
            self.qnet.update()

            if self.time_step % 1000 == 0:
                self.qnet.save_params(self.model_savefile)

            if self.time_step % update_time == 0:
                self.copy_target_network()

    def get_best_action(self, current_state):
        input_matrix = np.zeros((batch_size, self.channels, resolution[0], resolution[1]))
        input_matrix[0] = current_state
        self.target.forward(mx.io.DataBatch([mx.nd.array(input_matrix, ctx)],[]))
        qvalue = np.squeeze(self.target.get_outputs()[0].asnumpy()[0])
        action_index = np.argmax(qvalue)
        return action_index


if __name__ == '__main__':
    my_agent = AgentDQN(3, channels=3)
    print('My agent has initialized!')