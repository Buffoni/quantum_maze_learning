#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Methods on quantum evolution for quantum_maze_env

@author: Nicola Dalla Pozza
"""
import time
from qutip import *
from gym_quantum_maze.envs.maze_tools import *


def run_maze(adjacency_matrix, initial_quantum_state, sinkerNode, p, timeSamples, sink_rate=1., dt=1e-2):
    """ Run a simulation on the maze.

    Runs a simulation of the open quantum system described with a Lindblad equation obtained from the adjacency matrix.

    Reference: Caruso, Universally optimal noisy quantum walks on complex networks, New J. Phys. 16 055015 (2014)

    Parameters
    ----------
    adjacency_matrix : np.array, dtype=bool
        adjacency matrix with entry adjacency_matrix[i, j] for the node between j and i
    initial_quantum_state : qutip.qobj.Qobj
        quantum state of the system at the beginning of the simulation
    sinkerNode : int
        index of the sinker node
    p : float
        parameter in the Lindblad equation
    timeSamples :
        number of time samples considered in the time equation
    sink_rate : float (default 1.)
        transfer rate to the sink
    dt : float (default 10**-2)
        single step time interval

    Returns
    -------
    (qutip.Result.final_state, float)
        return the final quantum state at the end of the quantum simulation and the elapsed time for
        the simulation
    """
    start = time.time()
    N = adjacency_matrix.shape[0]
    D = np.sum(adjacency_matrix, axis=0)  # degree vector representing the diagonal of the degree matrix
    T = np.array([[adjacency_matrix[i, j] / D[j] if D[j] > 0 else 0 for i in range(N)] for j in range(N)])
    # normalized laplacian of the classical random walk
    S = np.zeros([N + 1, N + 1])  # transition matrix to the sink
    S[N, sinkerNode] = np.sqrt(2 * sink_rate)

    H = Qobj((1 - p) * np.pad(adjacency_matrix, [(0, 1), (0, 1)], 'constant'))
    # add zero padding to account for the sink and multiply for (1-p)
    L = [np.sqrt(p * T[i, j]) * (basis(N + 1, i) * basis(N + 1, j).dag())
         for i in range(N) for j in range(N) if T[i, j] > 0]  # set of Lindblad operators
    L.append(Qobj(S))  # add the sink transfer

    obs = [basis(N + 1, sinkerNode) * basis(N + 1, sinkerNode).dag()]  # site that we want to observe obs=|N><N|
    times = np.arange(1, timeSamples + 1) * dt  # timesteps of the evolution

    opts = Options(store_states=False, store_final_state=True)  # , nsteps=3000)
    result = mesolve(H, initial_quantum_state, times, L, obs, options=opts)  # solve master equation

    end = time.time()

    return result.final_state, end - start


def plot_maze_and_quantumState(maze, quantumState, show_nodes=False, show_links=False, show_ticks=False,
                               color_map='CMRmap_r', show=True):
    """Plot a maze together with a quantum state.

    Uses plot_maze to plot the maze, and on it superimposes the distribution of the diagonal entries of the quantum
     state defined on the nodes. White pixels (value 1) are paths while black pixels are walls. The quantum state is
     colored from blue (close to 1) to yellow (close to 0)

    Parameters
    ----------
    maze : Maze
        instance of class Maze that defines the adjacency matrix
    quantumState : qutip.qobj.Qobj
        quantum state to plot
    show_nodes : bool (default False)
        options to show the node number on the corresponding pixel in black
    show_links : bool (default False)
        options to show the link number on the corresponding pixel in red
    show_ticks : bool (default False)
        options to show the coordinate ticks in the plot
    color_map : matplotlib.pyplot (default viridis_r)
        color_map for quantum state
    show : bool (default False)
            options to show the plot or not

    Returns
    -------
    numpy.ndarray, AxesImage
        numpy array with rgb colors for each pixel, AxesImage obtained from pyplot.imshow() (when plotted)
    """

    img, ax = maze.plot_maze(show_nodes=show_nodes, show_links=show_links, show=False, show_ticks=show_ticks)

    diagQS = quantumState.diag()
    cmap = plt.cm.get_cmap(color_map)
    norm = plt.Normalize(0, 1)
    # img = cmap(np.zeros((2*maze.height - 1, 2*maze.width - 1)), alpha=0)
    for n in range(diagQS.size - 1):  # -1  because the last one is the sink
        # if diagQS[n] > 10*np.finfo(diagQS.dtype).eps:
        x, y = maze.node2xy(n)
        img[y, x, :] = cmap(norm(diagQS[n]))

    for link in range(1, maze.total_links + 1):
        x, y = maze.link2xy(link)
        if maze.get_link(link) > 0:
            if 1 <= link <= maze.vertical_links and not np.array_equal(img[y + 1, x, :], [1, 1, 1, 1]) \
                    and not np.array_equal(img[y - 1, x, :], [1, 1, 1, 1]):
                img[y, x, :] = (img[y + 1, x, :] + img[y - 1, x, :]) / 2
            elif maze.vertical_links < link <= maze.total_links and not np.array_equal(img[y, x + 1, :], [1, 1, 1, 1]) \
                    and not np.array_equal(img[y, x - 1, :], [1, 1, 1, 1]):
                img[y, x, :] = (img[y, x + 1, :] + img[y, x - 1, :]) / 2

    if maze.startNode is not None:
        x, y = maze.node2xy(maze.startNode)
        img[y, x, :3] = 0, 0, 1
        # plt.text(x - xshift, y - yshift, str(self.startNode), fontweight='bold') # always print startNode

    if maze.sinkerNode is not None:
        x, y = maze.node2xy(maze.sinkerNode)
        img[y, x, :3] = 1, 0, 0
        # plt.text(x - xshift, y - yshift, str(self.sinkerNode), fontweight='bold') # always print sinkerNode

    if show:
        ax = plt.imshow(img, origin='lower')
        plt.show()
    else:
        ax = None

    return img, ax


def run_maze_save_dynamics(adjacency_matrix, initial_quantum_state, sinkerNode, p, timeSamples, sink_rate=1., dt=1e-2,
                           quantum_states_saved=10):
    """ Run a simulation on the maze and save the sequence of quantum states.

    Runs a simulation of the open quantum system described with a Lindblad equation obtained from the adjacency matrix.

    Reference: Caruso, Universally optimal noisy quantum walks on complex networks, New J. Phys. 16 055015 (2014)

    Parameters
    ----------
    adjacency_matrix : np.array, dtype=bool
        adjacency matrix with entry adjacency_matrix[i, j] for the node between j and i
    initial_quantum_state : qutip.qobj.Qobj
        quantum state of the system at the beginning of the simulation
    sinkerNode : int
        index of the sinker node
    p : float
        parameter in the Lindblad equation
    timeSamples :
        number of time samples considered in the time equation
    sink_rate : float (default 1.)
        transfer rate to the sink
    dt : float (default 10**-2)
        single step time interval
    quantum_states_saved : int
        number of quantum state saved during time evolution

    Returns
    -------
    (list of qutip.Result.final_state, float)
        return the final quantum state at the end of the quantum simulation and the elapsed time for
        the simulation
    """
    start = time.time()

    quantum_state = initial_quantum_state
    result_list = []
    for k in range(quantum_states_saved):
        quantum_state, _ = run_maze(adjacency_matrix=adjacency_matrix, initial_quantum_state=quantum_state,
                                    sinkerNode=sinkerNode, p=p, timeSamples=(timeSamples // quantum_states_saved),
                                    sink_rate=sink_rate, dt=dt)
        result_list.append(quantum_state)

    end = time.time()

    return result_list, end - start


if __name__ == "__main__":
    myMaze = Maze(maze_size=(10, 5))
    quantum_system_size = myMaze.width * myMaze.height + 1
    myQuantumState = ket2dm(basis(quantum_system_size, myMaze.startNode))
    finalQuantumState, _ = run_maze(myMaze.adjacency, myQuantumState, myMaze.sinkerNode, 0.3, 1000)
    plot_maze_and_quantumState(myMaze, finalQuantumState)
