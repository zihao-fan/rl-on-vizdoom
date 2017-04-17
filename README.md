# rl-on-vizdoom

Deep Reinforcement Learning algorithms implementated using MxNet.

### A3C

An implementation of A3C on GPU. Use mini-batch instead of asynchronous updates.
To train the model, use:

    python train.py

in src_a3c.

### DQN

DQN with Replay Buffer and Target Network. Use part of the Inception-BN network as default feature extractor. 
To train this model, use:

    python train_dqn.py
