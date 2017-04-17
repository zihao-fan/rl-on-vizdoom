#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
import numpy as np
from random import sample, randint, random
import os

current_path = os.path.realpath(__file__)
root_path = '/'.join(current_path.split('/')[:-2])

def get_filenames_from_dir(my_dir):
    my_files = [ f for f in os.listdir(my_dir) if os.path.isfile(os.path.join(my_dir, f))]
    return my_files

class ReplayMemory:
    def __init__(self, capacity, resolution, s1_frames=1):
        state_in_shape = (capacity, s1_frames, resolution[0], resolution[1])
        state_out_shape = (capacity, s1_frames, resolution[0], resolution[1])
        self.s1 = np.zeros(state_in_shape, dtype=np.float32)
        self.s2 = np.zeros(state_out_shape, dtype=np.float32)
        self.a = np.zeros(capacity, dtype=np.int32)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.is_terminal = np.zeros(capacity, dtype=np.bool_)
        self.capacity = capacity
        self.size = 0
        self.pos = 0

    def add_transition(self, s1, action, s2, is_terminal, reward):
        self.s1[self.pos] = s1
        self.a[self.pos] = action
        if not is_terminal:
            self.s2[self.pos] = s2
        self.is_terminal[self.pos] = is_terminal
        self.r[self.pos] = reward

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def get_sample(self, sample_size):
        i = sample(range(0, self.size), sample_size)
        return self.s1[i], self.a[i], self.s2[i], self.is_terminal[i], self.r[i]

if __name__ == '__main__':
    memory = ReplayMemory(capacity=1000, resolution=(30, 40), s1_frames=1)
    print('Test succeed!')