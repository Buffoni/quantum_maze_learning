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
import os

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from ray import tune

from gym_quantum_maze.envs import quantum_maze_env

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

    def __init__(self, inputs, outputs, intermediates=None, env=None, diag_threshold=10**(-14)):
        self.inputs = inputs
        self.outputs = outputs
        self.env = env
        self.diag_threshold = diag_threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        super(DQN, self).__init__()
        if intermediates is None:
            intermediates = [int(np.sqrt(inputs * outputs)), int(np.sqrt(outputs * np.sqrt(inputs * outputs)))]
        if isinstance(intermediates, int):  # int to List conversion
            intermediates = [intermediates]

        intermediates.insert(0, inputs)
        intermediates.append(outputs)
        self.layers = nn.ModuleList()
        for k in range(len(intermediates) - 1):
            self.layers.append(nn.Linear(intermediates[k], intermediates[k + 1]))

    def forward(self, x):
        if self.env is not None:
            mask = torch.zeros(self.outputs, requires_grad=False, device=self.device)
            mask[0] = 1. # allows for 'no action' action
            for n in range(self.env.quantum_system_size-1):
                if x[0][n] >= self.diag_threshold:
                    nx, ny = self.env.maze.node2xy(n)
                    if nx > 0:
                        mask[self.env.maze.xy2link(nx - 1, ny)] = 1.
                    if nx < self.env.maze.width:
                        mask[self.env.maze.xy2link(nx + 1, ny)] = 1.
                    if ny > 0:
                        mask[self.env.maze.xy2link(nx, ny - 1)] = 1.
                    if ny < self.env.maze.height:
                        mask[self.env.maze.xy2link(nx, ny + 1)] = 1.
        else:
            mask = torch.ones(self.outputs, requires_grad=False, device=self.device)
        for single_layer in self.layers:
            x = F.relu(single_layer(x))
        return F.normalize(x*mask)

# deep_Q_learning
def deep_Q_learning_maze(maze_filename=None, p=0.1, time_samples=100, total_actions=4,
                         num_episodes=100, changeable_links=None,  # [4, 15, 30, 84],
                         batch_size=128, gamma=0.999, eps_start=0.9, eps_end=0.05,
                         eps_decay=1000, target_update=10, replay_capacity=512,
                         save_filename=None, enable_tensorboardX=True, enable_saving=True,
                         startNode=None, sinkerNode=None, training_startNodes=None,
                         action_selector=None, state_selector=3, diag_threshold=10**(-12), link_update=0.1, action_mode='reverse'):
    """Function that performs the deep Q learning.

    Reference: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

    action_selector:
        None (default), i.e. pick any action
        'threshold_mask', i.e. pick any action near a node with population greater than diag_threshold
        'probability_mask', i.e. pick a node according to the population, and from that an action according to the links
                            from that node
    """
    tic = time.time()
    codeName = 'deep_Q_learning_maze'
    today = datetime.datetime.now()

    env = gym.make('quantum-maze-v0', maze_filename=maze_filename, startNode=startNode, sinkerNode=sinkerNode,
                   p=p, sink_rate=1, time_samples=time_samples, changeable_links=changeable_links,
                   total_actions=total_actions, done_threshold=0.95, link_update=link_update, action_mode=action_mode,
                   state_selector=state_selector)
    if startNode is None:
        startNode = env.initial_maze.startNode
    if sinkerNode is None:
        sinkerNode = env.initial_maze.sinkerNode
    if training_startNodes is None or training_startNodes == []:
        training_startNodes = [startNode]

    # if gpu is to be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config_options = '_ST{0}_A{1}_T{2}_P{3:02.0f}'.format(state_selector, total_actions, time_samples, 10 * p)
    if save_filename is None and enable_saving:
        save_filename = ''.join((today.strftime('%Y-%m-%d_%H-%M-%S_'), codeName, config_options))

    if enable_tensorboardX:
        writer = SummaryWriter(os.path.join('tensorboardX', save_filename))  # tensorboardX writer

    # Get number of actions from gym action space
    n_actions = env.action_space.n

    # Setup Neural Network
    if action_selector == 'threshold_mask':
        policy_net = DQN(len(env.state), n_actions, env=env, diag_threshold=diag_threshold).to(device)
        target_net = DQN(len(env.state), n_actions, env=env, diag_threshold=diag_threshold).to(device)
    elif action_selector == 'probability_mask':
        policy_net = DQN(len(env.state), n_actions, env=env, diag_threshold=diag_threshold).to(device)
        target_net = DQN(len(env.state), n_actions, env=env, diag_threshold=diag_threshold).to(device)
    else:
        policy_net = DQN(len(env.state), n_actions).to(device)
        target_net = DQN(len(env.state), n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.RMSprop(policy_net.parameters())
    memory = ReplayMemory(replay_capacity)

    def select_action(state, eps_threshold, net=policy_net, noAction_probability=0.05):
        sample = random.random()
        if sample > eps_threshold:
            with torch.no_grad():
                return net(state).max(1)[1].view(1, 1)
        else:
            if action_selector == 'threshold_mask':
                mask = set()
                for n in range(env.quantum_system_size - 1):
                    if env.quantum_state.full()[n, n] >= diag_threshold:
                        nx, ny = env.maze.node2xy(n)
                        if nx > 0:
                            mask.add(env.maze.xy2link(nx - 1, ny))
                        if nx < env.maze.width:
                            mask.add(env.maze.xy2link(nx + 1, ny))
                        if ny > 0:
                            mask.add(env.maze.xy2link(nx, ny - 1))
                        if ny < env.maze.height:
                            mask.add(env.maze.xy2link(nx, ny + 1))
                return torch.tensor([[random.choice(tuple(mask))]], device=device, dtype=torch.long)
            elif action_selector == 'probability_mask':
                # note that in this way you always define a valid action, action=0 (noAction) is never selected
                population = np.real(np.diag(env.quantum_state.full())[:env.quantum_system_size - 1])
                if population.min() < 0:
                    population = population - population.min()
                population = population/population.sum()  # normalize
                n = np.random.choice(range(env.quantum_system_size - 1), 1, p=population)
                nx, ny = env.maze.node2xy(n)
                available_links = [0] # noAction
                if nx > 0:
                    available_links.append(1)
                if nx < env.maze.width:
                    available_links.append(3)
                if ny > 0:
                    available_links.append(2)
                if ny < env.maze.height:
                    available_links.append(4)
                selected_direction = np.random.choice(available_links, 1, p=[noAction_probability] + \
                                        [(1-noAction_probability)/(len(available_links)-1)]*(len(available_links)-1))
                if selected_direction == 1:
                    selected_link = env.maze.xy2link(nx - 1, ny)
                elif selected_direction == 3:
                    selected_link = env.maze.xy2link(nx + 1, ny)
                elif selected_direction == 2:
                    selected_link = env.maze.xy2link(nx, ny - 1)
                elif selected_direction == 4:
                    selected_link = env.maze.xy2link(nx, ny + 1)
                else:
                    selected_link = 0
                return torch.tensor([[selected_link]], device=device, dtype=torch.long)
            else:
                return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)

    episode_transfer_to_sink = []
    target_transfer_to_sink = []

    def optimize_model():
        if len(memory) < batch_size:
            return
        transitions = memory.sample(batch_size)

        batch = Transition(*zip(*transitions))

        if torch.__version__ < '1.2.0':
            non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device,
                                          dtype=torch.uint8) # version on qdab server
        else:
            non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device,
                                          dtype=torch.bool)  # current version, torch.uint8 is deprecated

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

    def evaluate_sequence_with_target_net(sequence=None):
        """ If sequence is None, evaluates the best action with target_net
        """
        if sequence is None:
            evaluate_optimal_action = True
            sequence = [0] * total_actions
        else:
            evaluate_optimal_action = False

        env.reset()
        state = torch.tensor(env.state, device=device, dtype=torch.float32).unsqueeze(0)
        episode_reward = 0
        for t in range(total_actions):
            # Select and perform an action
            if evaluate_optimal_action:
                eps_threshold = -1
                action = select_action(state, eps_threshold, target_net).item()
                sequence[t] = action
            else:
                action = sequence[t]

            next_state, reward, done, _ = env.step(action)
            # reward = reward.to(device='cpu') #torch.tensor([reward], device=device)
            episode_reward = reward + gamma * episode_reward

            if done:
                next_state = None
            else:
                next_state = torch.tensor(next_state, device=device, dtype=torch.float32).unsqueeze(0)

            # Move to the next state
            state = next_state

            if done:
                break

        return episode_reward, sequence

    # Training loop
    steps_done = 0
    for i_episode in range(num_episodes):
        episode_reward_startNode = [0] * len(training_startNodes)
        eps_threshold = eps_end + (eps_start - eps_end) * math.exp(-1. * steps_done / eps_decay)
        steps_done += 1
        for id_startNode, startNode_tmp in enumerate(training_startNodes):
            # Initialize the environment and state
            env.initial_maze.startNode = startNode_tmp
            env.reset()
            state = torch.tensor(env.state, device=device, dtype=torch.float32).unsqueeze(0)
            episode_reward = 0
            for t in range(total_actions):
                # Select and perform an action
                action = select_action(state, eps_threshold)
                next_state, reward, done, _ = env.step(action.item())
                reward = torch.tensor([reward], device=device)
                episode_reward = reward + gamma * episode_reward

                if done:
                    next_state = None
                else:
                    next_state = torch.tensor(next_state, device=device, dtype=torch.float32).unsqueeze(0)

                # Store the transition in memory
                memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state
                # if any(np.real(np.diag(env.quantum_state.full())) < 0):
                #     print(np.real(np.diag(env.quantum_state.full())))

                if done:
                    break

            # Perform one step of the optimization (on the target network)
            optimize_model()
            episode_reward_startNode[id_startNode] = episode_reward.to(device='cpu')
            episode_transfer_to_sink.append(episode_reward_startNode[id_startNode])
            tune.report(episode_reward=episode_reward_startNode[id_startNode])

        if enable_tensorboardX:
            if len(training_startNodes) > 1:
                writer.add_scalar('data/episode_average_reward',
                                  sum(episode_reward_startNode) / len(episode_reward_startNode),
                                  i_episode)  # tensorboardX log
                for id_startNode, startNode_tmp in enumerate(training_startNodes):
                    writer.add_scalar('data/episode_reward_startNode_{0:d}'.format(startNode_tmp),
                                      episode_reward_startNode[id_startNode], i_episode)
            else:
                writer.add_scalar('data/episode_reward', episode_reward.to(device='cpu'), i_episode)  # tensorboardX log

        # Update the target network, copying all weights and biases in DQN
        if i_episode % target_update == 0:
            env.initial_maze.startNode = startNode
            reward_target, _ = evaluate_sequence_with_target_net(None)
            if enable_tensorboardX:
                writer.add_scalar('data/target_reward', reward_target, i_episode)
            target_transfer_to_sink.extend([reward_target] * target_update)
            target_net.load_state_dict(policy_net.state_dict())
            # print('Completed episode', i_episode, 'of', num_episodes)

    env.initial_maze.startNode = startNode
    reward_no_actions, _ = evaluate_sequence_with_target_net([0] * total_actions)
    reward_final, optimal_sequence = evaluate_sequence_with_target_net(None)

    # Save variables:
    if enable_saving:
        save_variables(os.path.join('simulations', save_filename),
                       episode_transfer_to_sink, env, steps_done, policy_net, maze_filename, p, time_samples,
                       total_actions,
                       num_episodes, changeable_links, batch_size, gamma, eps_start, eps_end, eps_decay, target_update,
                       replay_capacity, reward_no_actions, reward_final, optimal_sequence, target_transfer_to_sink)

    toc = time.time()
    elapsed = toc - tic

    return save_filename, elapsed, reward_final, optimal_sequence


def save_variables(filename=None, *args):
    """ TODO: atm it saves only 1 net, solve to possible con with target_net, policy_net  """
    if filename is None:
        today = datetime.datetime.now()
        filename = today.strftime('%Y-%m-%d_%H-%M-%S')

    with open(filename + '.pkl', 'wb') as f:
        pickle.dump([x.to('cpu') if isinstance(x, torch.Tensor) else x for x in args if not isinstance(x, DQN)], f)

    for x in args:
        if isinstance(x, DQN):
            torch.save(x.state_dict(), filename + '_policy_net.pt')

    return filename


def plot_durations(episode_transfer_to_sink, title='Training...', constants=[], legend_labels=[]):
    plt.figure(dpi=300)
    plt.clf()
    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel('Transfer to Sink')

    training_legend_labels = []
    generate_training_labels = not len(episode_transfer_to_sink) + len(constants) == len(legend_labels)

    for (i, transfer_to_sink) in enumerate(episode_transfer_to_sink):
        # transfer_to_sink_t = torch.cat(transfer_to_sink)
        transfer_to_sink_t = torch.tensor(transfer_to_sink, dtype=torch.float)
        transfer_to_sink_len = len(transfer_to_sink_t)
        plt.plot(transfer_to_sink_t.numpy())
        # Take 100 episode averages and plot them too
        if transfer_to_sink_len >= 100:
            means = transfer_to_sink_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())
            if generate_training_labels:
                training_legend_labels.extend(['training-' + str(i), 'mean 100-' + str(i)])
            else:
                training_legend_labels.extend([legend_labels[i], legend_labels[i] + ' mean 100'])

        else:
            if generate_training_labels:
                training_legend_labels.append('training-' + str(i))
            else:
                training_legend_labels.append(legend_labels[i])

    if not constants == []:
        for k in constants:
            plt.plot([k] * transfer_to_sink_len)
        legend_labels = training_legend_labels + legend_labels[-len(constants):]
    else:
        legend_labels = training_legend_labels

    plt.legend(legend_labels)
    plt.ylim(0, 1)
    plt.show()

def tuneLauncher(conf):
    training_startNodes = []
    action_selector = 'probability_mask'  # None, 'threshold_mask', 'probability_mask'
    diag_threshold = 10 ** (-4)
    link_update = 0.1
    action_mode = 'reverse'  # 'reverse', 'sum', 'subtract'

    deep_Q_learning_maze(maze_filename='maze_8x8.pkl',
                         time_samples=conf['t_value-actions'][0], num_episodes=2000, p=conf['p_value'],
                         total_actions=conf['t_value-actions'][1],
                         training_startNodes=training_startNodes,
                         action_selector=action_selector,
                         diag_threshold=diag_threshold,
                         link_update=link_update,
                         action_mode=action_mode,
                         state_selector=conf['state_selector'],
                         )


if __name__ == '__main__':

    print('learning_tools has started')

    # config = {
    #     'state_selector': tune.grid_search([1, 3]),
    #     'p_value': tune.grid_search([0, 0.2, 0.4, 0.6, 0.8, 1]),
    #     't_value-actions': tune.grid_search([(500, 8), (1000, 8), (1500, 8), (2000, 8), (3000, 4), (1000, 12)]),
    # }

    config = {
        'state_selector': tune.grid_search([1]),
        'p_value': tune.grid_search([0]),
        't_value-actions': tune.grid_search([(500, 8)]),
    }

    trialResources = {'cpu': 1./10, 'gpu': 1./10}

    #Training section, uncomment on the server only!!!
    analysis = tune.run(tuneLauncher, config=config,
                        resources_per_trial=trialResources, local_dir='tuneOutput')
    print("BEST PARAMETERS")
    print(analysis.get_best_config(metric="episode_reward"))
    analysis.dataframe().to_pickle('tuneAnalysis.p')


    # This section prints the results of a trained agent at different p
    filename = ['P00', 'P02', 'P04', 'P06', 'P08', 'P10']
    final_trained = []
    final_untrained = []
    for p in filename:
        with open(os.path.join('new_simulations', 'deep_Q_learning_maze_ST1_A8_T1000_' + p + '.pkl'), 'rb') as f:
            [episode_transfer_to_sink, env, steps_done, maze_filename, p, time_samples, total_actions,
             num_episodes, changeable_links, batch_size, gamma, eps_start, eps_end, eps_decay, target_update,
             replay_capacity, reward_no_actions, reward_final, optimal_sequence, target_transfer_to_sink] = pickle.load(f)
            final_trained.append(reward_final)
            final_untrained.append(reward_no_actions)
    plt.plot(final_trained, label='Trained Agent')
    plt.plot(final_untrained,  label='No Actions')
    plt.legend()
    plt.show()
    #print(reward_no_actions)
    #print(reward_final)

    #This section prints one training
    with open(os.path.join('new_simulations', 'deep_Q_learning_maze_ST1_A8_T1000_P02.pkl'), 'rb') as f:
        [episode_transfer_to_sink, env, steps_done, maze_filename, p, time_samples, total_actions,
         num_episodes, changeable_links, batch_size, gamma, eps_start, eps_end, eps_decay, target_update,
         replay_capacity, reward_no_actions, reward_final, optimal_sequence, target_transfer_to_sink] = pickle.load(f)

    legend_labels=[]
    if len(training_startNodes) > 1:
        episode_transfer = [episode_transfer_to_sink[k::len(training_startNodes)] for k in
                            range(len(training_startNodes))]
        episode_transfer.append(
            [sum(episode_transfer_to_sink[k * len(training_startNodes):(k + 1) * len(training_startNodes)])
             / len(training_startNodes) for k in range(num_episodes)])
        legend_labels = ['tranining']*(len(training_startNodes)+1)
    else:
        episode_transfer = [episode_transfer_to_sink]
        training_startNodes = [0]
    episode_transfer.append(target_transfer_to_sink)
    legend_labels.append('target')

    constants = [reward_no_actions, reward_final]
    legend_labels.extend(['no action', 'final'])
    plot_durations(episode_transfer,
                   title=''.join(('p=', str(env.p), ', T=', str(env.time_samples))),
                   constants=constants,
                   legend_labels=legend_labels
                   )
