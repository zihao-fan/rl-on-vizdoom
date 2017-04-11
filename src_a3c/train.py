"""Trains an `Agent` using trajectories from multiple environments."""
from __future__ import print_function
import argparse
from itertools import chain
import time
import gym
import numpy as np
import mxnet as mx
from config import Config
from model import Agent
from vizdoom_wrapper import VizdoomWrapper

def train_episode(agent, envs, t_max):
    """Complete an episode's worth of training for each environment."""
    num_envs = len(envs)

    # Buffers to hold trajectories, e.g. `env_xs[i]` will hold the observations
    # for environment `i`.
    env_xs, env_as = _2d_list(num_envs), _2d_list(num_envs)
    env_rs, env_vs = _2d_list(num_envs), _2d_list(num_envs)
    episode_rs = np.zeros(num_envs, dtype=np.float)

    for env in envs:
        env.reset()

    observations = [e.get_current_state()[0] for e in envs]

    done = np.array([False for _ in range(num_envs)])
    all_done = False
    t = 1
    # NOTE(reed): For simplicity, it stops this episode when any of the envs is
    # done. As a side effect, the episode_rs may appear to vibrate for the
    # initial rounds instead of decreasing gradually.
    while not all_done:
        # NOTE(reed): Rebind to set the data shape.
        # Binded shape is Provided shape !!
        agent.model.bind(
            data_shapes=[('data', (num_envs,)+agent.input_size)],
            label_shapes=None,
            for_training=False,
            force_rebind=True,
            grad_req="null")

        # step_xs.shape == (num_envs,) + input.shape
        step_xs = np.vstack([o for o in observations])

        # Get actions and values for all environments in a single forward pass.
        step_xs_nd = mx.nd.array(step_xs, ctx=agent.ctx)
        data_batch = mx.io.DataBatch(data=[step_xs_nd], label=None)
        agent.model.forward(data_batch, is_train=False)
        _, step_vs, _, step_ps = agent.model.get_outputs()

        step_ps = step_ps.asnumpy()
        step_vs = step_vs.asnumpy()
        step_as = agent.act(step_ps)

        # Step each environment whose episode has not completed.
        for i, env in enumerate(envs):
            if not done[i]:
                r = env.make_action(step_as[i])
                obs = env.get_current_state()[0]
                done[i] = env.is_terminal()

                # Record the observation, action, value, and reward in the
                # buffers.
                env_xs[i].append(step_xs[i])
                env_as[i].append(step_as[i])
                env_vs[i].append(step_vs[i][0])
                env_rs[i].append(r)
                episode_rs[i] += r

                # Add 0 as the state value when done.
                if done[i]:
                    env_vs[i].append(0.0)
                else:
                    observations[i] = envs[i].get_current_state()[0]

        # Perform an update every `t_max` steps.
        if t == t_max: # and not any_done:
            # If the episode has not finished, add current state's value. This
            # will be used to 'bootstrap' the final return (see Algorithm S3
            # in A3C paper).
            step_xs = np.vstack([o for o in observations])
            step_xs_nd = mx.nd.array(step_xs, ctx=agent.ctx)
            data_batch = mx.io.DataBatch(data=[step_xs_nd], label=None)
            agent.model.forward(data_batch, is_train=False)
            _, extra_vs, _, _ = agent.model.get_outputs()
            extra_vs = extra_vs.asnumpy()
            for i in range(num_envs):
                if not done[i]:
                    env_vs[i].append(extra_vs[i][0])

            # Perform update and clear buffers.
            paralell_num = 0
            for idx in range(num_envs):
                paralell_num += len(env_xs[idx])
            env_xs = np.vstack(list(chain.from_iterable(env_xs))).reshape((-1,) + agent.input_size)
            agent.train_step(env_xs, env_as, env_rs, env_vs, paralell_num)
            env_xs, env_as = _2d_list(num_envs), _2d_list(num_envs)
            env_rs, env_vs = _2d_list(num_envs), _2d_list(num_envs)
            t = 0

        all_done = np.all(done)
        t += 1

    return episode_rs


def _2d_list(n):
    return [[] for _ in range(n)]


def save_params(save_pre, model, epoch):
    model.save_checkpoint(save_pre, epoch, save_optimizer_states=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-envs', type=int, default=16)
    parser.add_argument('--t-max', type=int, default=50)
    parser.add_argument('--env-type', default='PongDeterministic-v3')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--save-pre', default='checkpoints')
    parser.add_argument('--save-every', type=int, default=0)
    parser.add_argument('--num-episodes', type=int, default=100000)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--print-every', type=int, default=10)
    parser.add_argument('--gpu', action='store_true')

    # Parse arguments and setup configuration `config`
    args = parser.parse_args()
    config = Config(args)
    print('args=%s' % args)
    print('config=%s' % config.__dict__)

    envs = []
    # Create and seed the environments
    for i in range(config.num_envs):
        if i == 0:
            envs.append(VizdoomWrapper('simpler_basic.cfg', seed=i, reward_scale=0.01, display=True))
        else:
            envs.append(VizdoomWrapper('simpler_basic.cfg', seed=i, reward_scale=0.01))

    # envs = [VizdoomWrapper('simpler_basic.cfg', seed=i, reward_scale=0.01) for i in range(config.num_envs)]
    agent = Agent(envs[0].img_shape, envs[0].actions_num, config=config)
    # Train
    running_reward = None
    start = time.time()
    for i in range(args.num_episodes):
        tic = time.time()
        episode_rs = train_episode(agent, envs, t_max=config.t_max)

        for er in episode_rs:
            running_reward = er if running_reward is None else (
                0.99 * running_reward + 0.01 * er)

        if i % args.print_every == 0:
            print('Batch %d complete (%.2fs) (%.1fs elapsed) (episode %d), '
                  'batch avg. reward: %.2f, running reward: %.3f' %
                  (i, time.time() - tic, time.time() - start,
                   (i + 1) * args.num_envs, np.mean(episode_rs),
                   running_reward))

        if args.save_every > 0:
            if i % args.save_every == 0:
                save_params(args.save_pre, agent.model, i)