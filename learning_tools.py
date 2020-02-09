#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Set of functions implementing reinforcement algorithms on the maze environment

@author: Nicola Dalla Pozza
"""

import datetime
import math
import pickle
import random
import time
from collections import namedtuple
from itertools import count

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter

from gym_quantum_maze.envs import quantum_maze_env


def deep_Q_learning_maze(maze_filename=None, p=0.1, time_samples=100, total_actions=4,
                         num_episodes=100, changeable_links=None,  # [4, 15, 30, 84],
                         batch_size=128, gamma=0.999, eps_start=0.9, eps_end=0.05,
                         eps_decay=3000, target_update=10, replay_capacity=10000):
    """Function that performs the deep Q learning to be called in parallel fashion.

    Reference: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    """
    tic = time.time()
    codeName = 'deep_Q_learning_maze'
    today = datetime.datetime.now()

    env = gym.make('quantum-maze-v0', maze_filename=maze_filename, startNode=0, sinkerNode=38,
                   p=p, sink_rate=1, time_samples=time_samples, changeable_links=changeable_links,
                   total_actions=total_actions, done_threshold=0.95)

    # if gpu is to be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # tensorboardX writer
    config_options = '_EP{0}_A{1}_T{2}_P{3:02.0f}'.format(num_episodes, total_actions, time_samples, 10*p)
    writer = SummaryWriter(comment=config_options)

    # Replay Memory
    Transition = namedtuple('Transition',
                            ('state', 'action', 'next_state', 'reward'))

    class ReplayMemory(object):

        def __init__(self, capacity):
            self.capacity = capacity
            self.memory = []
            self.position = 0

        def push(self, *args):
            """Saves a transition."""
            if len(self.memory) < self.capacity:
                self.memory.append(None)
            self.memory[self.position] = Transition(*args)
            self.position = (self.position + 1) % self.capacity

        def sample(self, batch_size):
            return random.sample(self.memory, batch_size)

        def __len__(self):
            return len(self.memory)

    # Neural Network
    class DQN(nn.Module):

        def __init__(self, inputs, outputs):
            super(DQN, self).__init__()
            self.fc1 = nn.Linear(inputs, int(np.sqrt(inputs * outputs)))
            self.fc2 = nn.Linear(int(np.sqrt(inputs * outputs)), int(np.sqrt(outputs * np.sqrt(inputs * outputs))))
            self.fc3 = nn.Linear(int(np.sqrt(outputs * np.sqrt(inputs * outputs))), outputs)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            return self.fc3(x)

    # Get number of actions from gym action space
    n_actions = env.action_space.n

    # Setup Neural Network
    policy_net = DQN(len(env.state), n_actions).to(device)
    target_net = DQN(len(env.state), n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.RMSprop(policy_net.parameters())
    memory = ReplayMemory(replay_capacity)

    def select_action(state, steps_done):
        sample = random.random()
        eps_threshold = eps_end + (eps_start - eps_end) * math.exp(-1. * steps_done / eps_decay)
        if sample > eps_threshold:
            with torch.no_grad():
                return policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)

    episode_transfer_to_sink = []

    def optimize_model():
        if len(memory) < batch_size:
            return
        transitions = memory.sample(batch_size)

        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=device, dtype=torch.uint8)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(batch_size, device=device)
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * gamma) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        for param in policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()

    # Training loop
    steps_done = 0
    for i_episode in range(num_episodes):
        # Initialize the environment and state
        env.reset()
        state = torch.tensor(env.state, device=device).unsqueeze(0)
        episode_reward = 0
        for t in count():
            # Select and perform an action
            action = select_action(state, steps_done)
            steps_done += 1
            next_state, reward, done, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)
            episode_reward = reward + gamma * episode_reward

            if done:
                next_state = None
            else:
                next_state = torch.tensor(next_state, device=device).unsqueeze(0)

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            if done:
                break

        # Perform one step of the optimization (on the target network)
        optimize_model()
        episode_transfer_to_sink.append(episode_reward.to(device='cpu'))

        # tensorboardX log
        writer.add_scalar('data/episode_reward', episode_reward, i_episode)

        # Update the target network, copying all weights and biases in DQN
        if i_episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())
            # print('Completed episode', i_episode, 'of', num_episodes)

    variables_to_save = [episode_transfer_to_sink, env, steps_done]

    # Save variables:
    filename = ''.join((today.strftime('%Y-%m-%d_%H-%M-%S_'), codeName, config_options))
    with open(''.join((filename, '.pkl')), 'wb') as f:
        pickle.dump(variables_to_save, f)

    torch.save(policy_net.state_dict(), ''.join((filename, '_policy_net', '.pt')))

    toc = time.time()
    elapsed = toc - tic

    return filename, elapsed


def plot_durations(episode_transfer_to_sink, title='Training...'):
    plt.figure()
    plt.clf()
    transfer_to_sink_t = torch.tensor(episode_transfer_to_sink, dtype=torch.float)
    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel('Transfer to Sink')
    plt.plot(transfer_to_sink_t.numpy())
    # Take 100 episode averages and plot them too
    if len(transfer_to_sink_t) >= 100:
        means = transfer_to_sink_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())
    plt.show()

if __name__ == '__main__':
    filename, elapsed = deep_Q_learning_maze(time_samples=500, num_episodes=10)
    print('Variables saved in', ''.join((filename, '.pkl')))
    print('Trained model saved in', ''.join((filename, '_policy_net', '.pt')))
    print("Elapsed time", elapsed, "sec.\n")
    with open(''.join((filename, '.pkl')), 'rb') as f:
        [episode_transfer_to_sink, env, steps_done] = pickle.load(f)
    plot_durations(episode_transfer_to_sink, title=''.join(('p=', str(env.p), ', T=', str(env.time_samples))))