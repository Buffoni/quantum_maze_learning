#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quantum Maze Environment.

Version with
- discrete actions: it is possible to change the value of some links in the maze
- continuous state space, composed by the adjacency matrix, the full quantum state and the value of the taken action

@author: Nicola Dalla Pozza

TODO: implement render /render_video with gym.viewer
"""
import copy

import gym
from gym import spaces
from gym import utils
from matplotlib import animation
from mpl_toolkits.axes_grid1 import make_axes_locatable

from gym_quantum_maze.envs.quantum_tools import *
import os
import pickle
import copy


class QuantumMazeEnv(gym.Env, utils.EzPickle):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, maze_size=(10, 5), startNode=0, sinkerNode=None, p=0.1, sink_rate=1, time_samples=100,
                 changeable_links=None, total_actions=8, done_threshold=0.95, maze_filename=None):
        # evolution parameters
        self.time_samples = time_samples
        self.sink_rate = sink_rate
        self.p = p

        if maze_filename is None:
            self.initial_maze = Maze(maze_size=maze_size, startNode=startNode, sinkerNode=sinkerNode)
        else:
            self.initial_maze = Maze().load(maze_filename)

            # overwrite startNode and sinkerNode
            if startNode is not None:
                self.initial_maze.startNode = startNode
            if sinkerNode is not None:
                self.initial_maze.sinkerNode = sinkerNode

        self.quantum_system_size = self.initial_maze.total_nodes + 1  # +1 for the sink
        self.initial_quantum_state = ket2dm(basis(self.quantum_system_size, self.initial_maze.startNode))

        if changeable_links is not None:
            self.changeable_links = changeable_links
        else:
            self.changeable_links = np.arange(1, self.initial_maze.total_links + 1).tolist()  # all links are changeable

        self.total_actions = total_actions
        self.done_threshold = done_threshold

        # initial condition, updated in self.reset()
        self.quantum_state = None
        self.maze = None
        self.actions_taken = None
        self.reset()
        self.figure = None

        # action definition:
        # 0 = no action; n=1,... reverse link changeable_links(n)
        self.action_space = spaces.Discrete(1 + len(self.changeable_links))

        # observation_space definition: to define accordingly to the state definition
        self.observation_space = spaces.Box(shape=(self.quantum_system_size ** 2 + len(self.changeable_links) + 1,),
                                            low=0, high=1, dtype=np.float64)

        # provides functionality to save/load via pickle
        utils.EzPickle.__init__(self, maze_size, startNode, sinkerNode, p, sink_rate, time_samples, changeable_links,
                                total_actions, done_threshold, maze_filename)
        # TODO: not sure if it's the correct way to use EzPickle because 1) there is no documentation 2) only the
        #  initial configuration is saved, for instance not the last state


    @property
    def state(self):
        density_matrix = self.quantum_state.full()
        return [np.real(density_matrix[n, n]) for n in range(self.quantum_system_size)] + \
               [func(density_matrix[m, n]) for m in range(3) for n in range(m + 1, 3)
                for func in (lambda x: np.real(x), lambda x: np.imag(x))] + \
               [self.maze.get_link(link) for link in self.changeable_links] + \
               [self.actions_taken / self.total_actions]
        # action_taken is normalized. Note that this definition has a list of mixed types

    @state.setter
    def state(self, value):
        try:
            value1, value2, value3 = value
        except ValueError:
            raise ValueError("Pass an iterable with three items")
        else:
            self.quantum_state = value1
            self.maze = value2
            self.actions_taken = value3

    def reset(self):
        """Resets the state of the environment and returns an initial observation."""
        self.quantum_state = copy.deepcopy(self.initial_quantum_state)
        self.maze = copy.deepcopy(self.initial_maze)
        self.actions_taken = 0
        return self.state

    def do_action(self, action: int, _maze=None):
        """Perform an action, that is, set the value of the adjacency matrix"""
        if _maze is None:
            _maze = self.maze
        if 1 <= action <= len(self.changeable_links):
            _maze.reverse_link(self.changeable_links[int(action) - 1])
            # minus 1 to correctly index changeable_links.
            # action==0 is no action

    def is_done(self):
        """Check whether the game is finished

         Returns
        -------
        (bool)
            True if the portions of quantum state in the sink is equal or above self.done_threshold, False otherwise
        """
        return np.real(self.quantum_state.full()[self.quantum_system_size - 1, self.quantum_system_size - 1]) >= \
               self.done_threshold or self.actions_taken == self.total_actions

    def step(self, action: int):
        """Evaluate the transition to a new state and the reward from the current state with an action.

        Parameters
        ----------
        action : numpy.float64
            value to set to the link in the adjacency matrix.

        Returns
        -------
        state, float, bool, {}
            returns the new state, the reward, whether the evolution is done, {}
        """
        # update maze
        self.do_action(action)

        # update number of actions
        self.actions_taken += 1

        new_quantum_state, _ = run_maze(self.maze.adjacency, self.quantum_state,
                                        self.initial_maze.sinkerNode, self.p, self.time_samples, self.sink_rate)

        # reward defined as how much gets to the sink in the current state transition
        sink = self.quantum_system_size - 1
        reward = np.real(new_quantum_state.full()[sink, sink]) - np.real(self.quantum_state.full()[sink, sink])

        # update quantum state
        self.quantum_state = new_quantum_state

        self.state = (self.quantum_state, self.maze, self.actions_taken)

        return self.state, reward, self.is_done(), {}

    def render(self, show_nodes=False, show_links=False, show_ticks=False, color_map='CMRmap_r'):
        """Renders the environment.

        The set of supported modes varies per environment. (And some environments do not support rendering at all.)
        By convention, if mode is:
        - human: render to the current display or terminal and return nothing. Usually for human consumption.
        - rgb_array: Return an numpy.ndarray with shape (x, y, 3), representing RGB values for an x-by-y pixel image,
            suitable for turning into a video.

        Parameters
        ----------
        show_nodes : bool (default False)
            options to show the node number on the corresponding pixel in black
        show_links : bool (default False)
            options to show the link number on the corresponding pixel in red
        show_ticks : bool (default False)
            options to show the coordinate ticks in the plot
        color_map : matplotlib.colors
            palette used to color the quantum state in the maze

        Returns
        -------
        numpy.ndarray, AxesImage
            numpy array with rgb colors for each pixel, AxesImage obtained from pyplot.imshow() (when plotted)
        """
        img, ax = plot_maze_and_quantumState(self.maze, self.quantum_state, show_nodes=show_nodes,
                                             show_links=show_links, show_ticks=show_ticks, color_map=color_map)

        # TODO: make the environment use a single figure. Possible code (not working)
        # TODO: manage figure closing (otherwise memory stays occupied)
        # if self.figure is None:
        #     self.figure = plt.figure()
        # img, _ = plot_maze_and_quantumState(self.maze, self.quantum_state, show_nodes=show_nodes,
        #                                     show_links=show_links, show_ticks=show_ticks, color_map=color_map)
        #
        # plt.figure(self.figure.number)
        # ax = plt.imshow(img)
        # plt.show()
        # TODO: show_nodes = True not working

        return img, ax

    def render_video(self, file_name='quantum_maze_video_test', frames_per_evolution=10, action_sequence=None,
                     color_map='CMRmap_r'):
        """Generates a video of the quantum system dynamics.

        The set of supported modes varies per environment. (And some environments do not support rendering at all.)
        By convention, if mode is:
        - human: render to the current display or terminal and return nothing. Usually for human consumption.
        - rgb_array: Return an numpy.ndarray with shape (x, y, 3), representing RGB values for an x-by-y pixel image,
            suitable for turning into a video.

        Parameters
        ----------
        file_name : str
            name used to save the animation (omit file extension)
        frames_per_evolution : int
            number of frame used in each time evolution
        action_sequence : list
            list of actions
        color_map : matplotlib.colors
            palette used to color the quantum state in the maze

        Returns
        -------
        numpy.ndarray, AxesImage
            numpy array with rgb colors for each pixel, AxesImage obtained from pyplot.imshow() (when plotted)
        """
        if action_sequence is None:
            action_sequence = []
        fig = plt.figure()
        ax = fig.add_subplot(111)
        div = make_axes_locatable(ax)
        cax = div.append_axes('right', '5%', '5%')

        _maze = copy.deepcopy(self.initial_maze)
        quantum_state_sequence = [self.initial_quantum_state] + \
                                 run_maze_save_dynamics(_maze.adjacency, self.initial_quantum_state,
                                                        _maze.sinkerNode, self.p, self.time_samples,
                                                        self.sink_rate, quantum_states_saved=frames_per_evolution)[0]

        frames = [plot_maze_and_quantumState(_maze, qs, show=False, color_map=color_map)[0]
                  for qs in quantum_state_sequence]

        for idx in range(len(action_sequence)):
            self.do_action(action=action_sequence[idx], _maze=_maze)
            quantum_state_sequence, _ = run_maze_save_dynamics(_maze.adjacency, quantum_state_sequence[-1],
                                                               _maze.sinkerNode, self.p,
                                                               self.time_samples, self.sink_rate,
                                                               quantum_states_saved=frames_per_evolution)
            frames += [plot_maze_and_quantumState(_maze, qs, show=False, color_map=color_map)[0]
                       for qs in quantum_state_sequence]

        im = ax.imshow(frames[0], origin='lower', cmap=color_map)  # Here make an AxesImage rather than contour
        fig.colorbar(im, cax=cax)

        ani = animation.FuncAnimation(fig, lambda x: im.set_data(frames[x]), frames=len(frames), interval=100,
                                      repeat_delay=1000, repeat=True)

        ani.save(''.join([file_name, '.mp4']))

        return ani

    def close(self):
        pass

    def seed(self, seed=None):
        pass


if __name__ == '__main__':
    print('quantum_maze_env has started')
    env = QuantumMazeEnv(time_samples=2000)
    env.reset()
    env.render()
    #
    env.step(action=1)
    env.render()
    #
    env.step(action=1)
    env.render()
    #
    env.render_video(action_sequence=[1, 2, 3, 4], file_name=os.path.join(os.path.curdir, '..\..\quantum_maze_video'))
    env.close()
