# -*- coding: utf-8 -*-
'''
Adopted from https://github.com/ebonyclock/deep_rl_vizdoom.git
'''
from __future__ import print_function
import numpy as np
import itertools as it
import vizdoom as vzd
import os
import cv2
import time

class VizdoomWrapper(object):
    '''
        This class helps with user to have APIs similar to gym.
        It also handles preprocessing, reward scaling and keeps track of recent frames.
    '''
    def __init__(self,
                 config_file,
                 frame_skip=4,
                 seed = 0,
                 display=False,
                 resolution=(80, 80),
                 stack_n_frames=4,
                 reward_scale=1.0,
                 noinit=False,
                 use_freedoom=False,
                 input_n_last_actions=False,
                 use_misc=False,
                 misc_scale=None,
                 scenarios_path=os.path.join(vzd.__path__[0], "scenarios"),
                 **kwargs):
        doom = vzd.DoomGame()

        if use_freedoom:
            doom.set_doom_game_path(vzd.__path__[0] + "/freedoom2.wad")

        doom.load_config(os.path.join(scenarios_path, str(config_file)))
        doom.set_window_visible(display)
        # TODO support for colors
        doom.set_screen_format(vzd.ScreenFormat.GRAY8)
        doom.set_screen_resolution(vzd.ScreenResolution.RES_320X240)
        doom.set_seed(seed)
        if not noinit:
            doom.init()
        self.doom = doom

        self._stack_n_frames = stack_n_frames
        assert len(resolution) == 2
        self._resolution = tuple(resolution)
        self._frame_skip = frame_skip
        self._reward_scale = reward_scale

        self._img_channels = stack_n_frames
        self.img_shape = (stack_n_frames, resolution[1], resolution[0])
        self.obs_size = stack_n_frames * resolution[1] * resolution[0]

        # TODO allow continuous actions
        # Allowing all combinations of action. If the possible action number is 3, then the size of the
        # action space will be 2^3. This could somehow solve the problem that policy peaks at one action.
        self._actions = [list(a) for a in it.product([0, 1], repeat=len(doom.get_available_buttons()))]
        self.actions_num = len(self._actions)
        self._current_screen = None
        self._current_stacked_screen = np.zeros(self.img_shape, dtype=np.float32)

        # Some algorithms may use these variables. Like the misc info (health, ammo, etc.) or the last n actions taken.
        # These part havn't been thoroughly tested.
        self._current_stacked_misc = None
        self.input_n_last_actions = 0
        self.misc_scale = None
        self.last_n_actions = None

        gvars_misc_len = len(doom.get_available_game_variables())
        if use_misc and (gvars_misc_len or input_n_last_actions):
            if misc_scale:
                assert len(misc_scale) <= gvars_misc_len
                self.misc_scale = np.ones(gvars_misc_len, dtype=np.float32)
                self.misc_scale[0:len(misc_scale)] = misc_scale

            self.misc_len = gvars_misc_len * self._stack_n_frames
            if input_n_last_actions:
                self.input_n_last_actions = input_n_last_actions
                self.last_n_actions = np.zeros(self.input_n_last_actions * self.actions_num, dtype=np.float32)
                self.misc_len += len(self.last_n_actions)

            self._current_stacked_misc = np.zeros(self.misc_len, dtype=np.float32)
            self.use_misc = True
        else:
            self.misc_len = 0
            self.use_misc = False

        if not noinit:
            self.reset()

    def _update_screen(self):
        self._current_screen = self.preprocess(self.doom.get_state().screen_buffer)
        self._current_stacked_screen = np.append(self._current_stacked_screen[1:], self._current_screen,
                                                 axis=0)

    def _update_misc(self):
        # TODO add support for input_n_actions without game variables
        game_vars = self.doom.get_state().game_variables
        if self.misc_scale:
            game_vars *= self.misc_scale
        if self.input_n_last_actions:
            game_vars_end_i = -len(self.last_n_actions)
        else:
            game_vars_end_i = len(self._current_stacked_misc)
        self._current_stacked_misc[0:len(game_vars) * (self._stack_n_frames - 1)] = self._current_stacked_misc[
                                                                                    len(game_vars):game_vars_end_i]
        self._current_stacked_misc[len(game_vars) * (self._stack_n_frames - 1):game_vars_end_i] = game_vars
        if self.input_n_last_actions:
            self._current_stacked_misc[-len(self.last_n_actions):] = self.last_n_actions

    def preprocess(self, img):
        # TODO check what's the difference in practice
        # img = cv2.resize(img, self._resolution, interpolation=cv2.INTER_CUBIC)
        # img = cv2.resize(img, self._resolution, interpolation=cv2.INTER_NEAREST)
        img = cv2.resize(img, self._resolution, interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0
        img = img.reshape([1] + list(img.shape))
        return img

    def reset(self):
        self.doom.new_episode()
        self._current_stacked_screen = np.zeros_like(self._current_stacked_screen)
        self._update_screen()

        if self.use_misc:
            if self.input_n_last_actions:
                self.last_n_actions.fill(0)
            self._current_stacked_misc = np.zeros_like(self._current_stacked_misc)
            self._update_misc()

    def make_action(self, action_index):
        '''
            take in a integer index (0-indexing)
            make the action and update the internal states
        '''
        action = self._actions[action_index]
        reward = self.doom.make_action(action, self._frame_skip) * self._reward_scale
        if not self.doom.is_episode_finished():
            if self.input_n_last_actions:
                self.last_n_actions[0:-self.actions_num] = self.last_n_actions[self.actions_num:]
                last_action = np.zeros(self.actions_num, dtype=np.int8)
                last_action[action_index] = 1
                self.last_n_actions[-self.actions_num:] = last_action

            self._update_screen()
            if self.use_misc:
                self._update_misc()
        return reward

    def get_current_state(self):
        '''
            returns 
                screen: 4D ndarray (1, frame_channel, width, height)
                misc: 2D ndarray (1, game variable num * channels)
        '''
        if self.use_misc:
            current_stacked_misc = np.asarray(self._current_stacked_misc, dtype=np.float32)
            return self._current_stacked_screen.reshape((1,) + self._current_stacked_screen.shape), \
                   current_stacked_misc.reshape((1,) + current_stacked_misc.shape)
        else:
            return self._current_stacked_screen.reshape((1,) + self._current_stacked_screen.shape), None

    def get_total_reward(self):
        return self.doom.get_total_reward() * self._reward_scale

    def is_terminal(self):
        return self.doom.is_episode_finished()

    def close(self):
        self.doom.close()

if __name__ == '__main__':
    doom = VizdoomWrapper(config_file='basic.cfg', display=False, use_misc=True, frame_skip=4)
    actions_num = doom.actions_num
    while not doom.is_terminal():
        screen, misc = doom.get_current_state()
        reward = doom.make_action(np.random.choice(np.arange(actions_num)))
        time.sleep(0.1)