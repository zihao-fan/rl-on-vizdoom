#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
from vizdoom import *
import numpy as np
from random import sample, randint, random
import skimage.color, skimage.transform
import itertools as it
from tqdm import trange
from time import time, sleep
import json, os

# algorithm relevant
from brain_dqn import BrainDQN
from brain_dqn import frame_repeat, resolution

# Training regime
epochs = 10
learning_steps_per_epoch = 20000#2000
test_episodes_per_epoch = 10
feature_extractor_list = ['pool_2_output', 'ch_concat_3c_chconcat_output',
'ch_concat_4e_chconcat_output', 'ch_concat_5b_chconcat_output']
# channel_number
channel_number = 3

# Other parameters
episodes_to_watch = 10

# config_file_path = '../scenarios/simpler_basic.cfg'
config_file_path = '../scenarios/cig.cfg'

def preprocess_resize(image, channels=3):
    '''
        (heigh, width, channel) -> (channel, height, width)
    '''
    if image is None:
        return None
    img = np.swapaxes(image, 0, 2)
    img = np.swapaxes(img, 1, 2)
    img = skimage.transform.resize(img, (channels, resolution[0], resolution[1]))
    img = img.astype(np.float32)
    return img

def perform_learning_step(brain, screen_buffer, epoch):
        '''
        Makes an action according to eps-greedy policy, observes the result 
        (next state, reward) and learns from the transition
        '''
        def exploration_rate(epoch):
            start_eps = 1.0
            end_eps = 0.1
            const_eps_epochs = 0.1 * epochs
            eps_decay_epochs = 0.6 % epochs

            if epoch < const_eps_epochs:
                return start_eps
            elif epoch < eps_decay_epochs:
                return start_eps - (epoch - const_eps_epochs) / \
                        (eps_decay_epochs - const_eps_epochs) * (start_eps - end_eps)
            else:
                return end_eps

        s1 = preprocess_resize(screen_buffer)

        eps = exploration_rate(epoch)
        if random() < eps:
            a = randint(0, brain.available_action_count - 1)
        else:
            a = brain.get_best_action(s1)
        reward = game.make_action(brain.actions[a], frame_repeat)

        is_terminal = game.is_episode_finished()
        s2 = preprocess_resize(game.get_state().screen_buffer) if not is_terminal else None

        brain.memory.add_transition(s1, a, s2, is_terminal, reward)
        brain.learn_from_memory()

def initialize_vizdoom(config_file_path):
    print('Initializing doom...')
    game = DoomGame()
    game.load_config(config_file_path)
    game.set_window_visible(True)
    game.set_mode(Mode.PLAYER)
    game.set_screen_format(ScreenFormat.RGB24)
    game.set_screen_resolution(ScreenResolution.RES_320X240)
    # cig configs
    game.add_game_args("-host 1 -deathmatch +timelimit 1.0 "
                   "+sv_forcerespawn 1 +sv_noautoaim 1 +sv_respawnprotect 1 +sv_spawnfarthest 1")
    game.add_game_args("+name AI +colorset 0")
    return game

game = initialize_vizdoom(config_file_path)
n = game.get_available_buttons_size()
bots = 7

def new_episode_with_bots(game):
    game.send_game_command("removebots")
    for i in range(bots):
        game.send_game_command("addbot")
    game.new_episode()
    return game

# ## -------------------------- training process --------------------------
extractor = feature_extractor_list[0]
result_dump_path = os.path.join('..', 'results', extractor + '.json')
game.init()
print('Doom initialized.')
my_brain = BrainDQN(n, feature_extractor=extractor, state_frames=channel_number)
print('Starting the training!')
time_start = time()
result_record = []
for epoch in range(epochs):
    print('\nEpoch %d\n-------' % (epoch + 1))
    train_episodes_finished = 0
    train_scores = []

    print('Training...')
    game = new_episode_with_bots(game)
    for learning_step in trange(learning_steps_per_epoch):
        screen_buffer = game.get_state().screen_buffer
        perform_learning_step(my_brain, screen_buffer, epoch)
        if game.is_episode_finished():
            score = game.get_total_reward()
            train_scores.append(score)
            game = new_episode_with_bots(game)
            train_episodes_finished += 1
        else:
            if game.is_player_dead():
                game.respawn_player()

    print('%d training episodes played.' % train_episodes_finished)

    train_scores = np.array(train_scores)

    print("Results: mean: %.1f±%.1f," % (train_scores.mean(), train_scores.std()), \
          "min: %.1f," % train_scores.min(), "max: %.1f," % train_scores.max())

    print('\nTesting...')
    test_episode = []
    test_scores = []
    for test_episode in trange(test_episodes_per_epoch):
        game = new_episode_with_bots(game)
        while not game.is_episode_finished():
            state = preprocess_resize(game.get_state().screen_buffer)
            best_action_index = my_brain.get_best_action(state)
            game.make_action(my_brain.actions[best_action_index], frame_repeat)
            if game.is_player_dead():
                game.respawn_player()
        r = game.get_total_reward()
        test_scores.append(r)

    test_scores = np.array(test_scores)
    test_scores_mean = test_scores.mean()
    test_scores_std = test_scores.std()
    test_scores_min = test_scores.min()
    test_scores_max = test_scores.max()
    print("Results: mean: %.1f±%.1f," % (test_scores_mean, test_scores_std), 
        "min: %.1f" % test_scores_min, "max: %.1f" % test_scores_max)

    total_time = (time() - time_start) / 60.0
    print('Total elapsed time: %.2f minutes.' % total_time)
    result_record.append((total_time, test_scores_mean, test_scores_std, test_scores_min, test_scores_max))

game.close()

with open(result_dump_path, 'w') as f:
    json.dump(result_record, f)
# print('======================================')
## -------------------------- training process --------------------------
# model_savefile = os.path.join('..', 'saved_weights', extractor+'.weights')
# print('Training finished. It\'s time to watch!')
# print('Loading the network weights from', model_savefile)
# my_brain = BrainDQN(n, feature_extractor=extractor, state_frames=channel_number, param_file=model_savefile)
# game = initialize_vizdoom(config_file_path)
# game.set_window_visible(True)
# game.set_mode(Mode.ASYNC_PLAYER)
# game.init()

# for _ in range(episodes_to_watch):
#     game.new_episode()
#     while not game.is_episode_finished():
#         state = preprocess_resize(game.get_state().screen_buffer)
#         best_action_index = my_brain.get_best_action(state)
#         game.set_action(my_brain.actions[best_action_index])
#         for _ in range(frame_repeat):
#             game.advance_action()

#     sleep(1.0)
#     score = game.get_total_reward()
#     print('Total score: ', score)