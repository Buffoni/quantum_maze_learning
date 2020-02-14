#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Maze class for quantum_maze_env

@author: Nicola Dalla Pozza
"""
import numpy as np
import matplotlib.pyplot as plt
from random import randrange, shuffle
import pickle
import datetime


def generate_adjacency_matrix(w, h):
    """Generate the adjacency matrix of a perfect maze given its width and height.

    Returns a np.array representing the adjacency matrix of a perfect maze with given width and height.
    adjacency[m, n] = 1 if there is a link between node m an n, 0 otherwise.
    The adjacency matrix is symmetric.

    Reference for the algorithm: https://rosettacode.org/wiki/Maze_generation#Python

    Parameters
    ----------
    w : int
        number of nodes in the x axis, i.e., the width of the maze
    h : int
        number of nodes in the y axis, i.e., the height of the maze

    Returns
    -------
    np.array, dtype='float64'
        symmetric adjacency matrix with entry A[i, j] = 1 if there is a link between node i and j, 0 otherwise
    """
    adjacency = np.zeros([h * w, h * w], dtype='float64')
    visited = [[0] * w + [1] for _ in range(h)] + [[1] * (w + 1)]

    def pos2node(x, y):
        return int(x * h + y)

    def walk(x, y):
        visited[y][x] = 1

        d = [(x - 1, y), (x, y + 1), (x + 1, y), (x, y - 1)]
        shuffle(d)
        for (xx, yy) in d:
            if visited[yy][xx]:
                continue
            if xx == x:
                adjacency[pos2node(x, yy), pos2node(x, y)] = 1
                adjacency[pos2node(x, y), pos2node(x, yy)] = 1
            if yy == y:
                adjacency[pos2node(x, y), pos2node(xx, y)] = 1
                adjacency[pos2node(xx, y), pos2node(x, y)] = 1

            walk(xx, yy)

    walk(randrange(w), randrange(h))

    return adjacency


class Maze(object):

    def __init__(self, adjacency=None, maze_size=(20, 10), startNode=0, sinkerNode=None):
        self.maze_size = maze_size
        self.width = int(self.maze_size[0])
        self.height = int(self.maze_size[1])

        if adjacency is None:
            self.adjacency = generate_adjacency_matrix(self.width, self.height)
        else:
            assert self.maze_size == adjacency.shape
            self.adjacency = adjacency

        self.total_nodes = self.width * self.height
        self.startNode = startNode
        self.sinkerNode = sinkerNode

        self.vertical_links = (self.height - 1) * self.width
        self.horizontal_links = (self.width - 1) * self.height
        self.total_links = self.vertical_links + self.horizontal_links

    @property
    def startNode(self):
        return self._startNode

    @startNode.setter
    def startNode(self, value):
        if value is None:
            self._startNode = 0
        else:
            assert 0 <= value < self.total_nodes, "startNode is outside the node range"
            self._startNode = value

    @property
    def sinkerNode(self):
        return self._sinkerNode

    @sinkerNode.setter
    def sinkerNode(self, value):
        if value is None:
            self._sinkerNode = self.width * self.height - 1
        else:
            assert 0 <= value < self.total_nodes, "sinkerNode is outside the node range"
            self._sinkerNode = value

    def generate_maze_map(self):
        """Generate the maze map representing the maze as a pixel image from its adjacency matrix.

        The map has value 1 if a link is present (the corresponding coefficient in the adjacency matrix is > 0) while 0
        represents a wall. The size of the map is (maze.height * 2 - 1, maze.width * 2 - 1). There are no border walls.

        Returns
        -------
        np.array, dtype=int
            Matrix representing the maze as a pixel image. Value 1 represents a link, value 0 represents
            a wall. Size of the matrix (maze.height * 2 - 1, maze.width * 2 - 1).

        """
        maze_map = np.zeros([2 * self.height - 1, 2 * self.width - 1], dtype='int')
        for j in range(self.width - 1):  # defines the upper triangular part of the adjacency matrix
            for i in range(self.height - 1):
                if self.adjacency[j * self.height + i, j * self.height + i + 1] > 0:
                    maze_map[2 * i + 1, 2 * j] = 1
                if self.adjacency[j * self.height + i, (j + 1) * self.height + i] > 0:
                    maze_map[2 * i, 2 * j + 1] = 1
            i = self.height - 1  # last node has no upper neighbour
            if self.adjacency[j * self.height + i, (j + 1) * self.height + i] > 0:
                maze_map[2 * i, 2 * j + 1] = 1

        j = self.width - 1  # last node has no right neighbour
        for i in range(self.height - 1):
            if self.adjacency[j * self.height + i, j * self.height + i + 1] > 0:
                maze_map[2 * i + 1, 2 * j] = 1

        for n in range(self.width * self.height):
            x, y = self.node2xy(n)
            maze_map[y, x] = 1

        return maze_map

    def plot_maze(self, show_nodes=False, show_links=False, show_ticks=False, show=True):
        """Plot the map of the maze. White pixels (value 1) are paths while black pixels are walls.

        Plots a maze with different options, such as the possibility to plot the number corresponding to each link
        (in red) or node (in black bold) with the show_links and show_nodes option. Option startNode and
        sinkerNode allow to insert the number of the start node or the sinker node, which are plotted in blue and in
        red respectively.

        Parameters
        ----------
        show_nodes : bool (default False)
            options to show the node number on the corresponding pixel in black
        show_links : bool (default False)
            options to show the link number on the corresponding pixel in red
        show_ticks : bool (default False)
            options to show the coordinate ticks in the plot
        show : bool (default False)
            options to show the plot or not


        Returns
        -------
        numpy.ndarray, AxesImage
            numpy array with rgb colors for each pixel, AxesImage obtained from pyplot.imshow() (when plotted)
        """
        maze_map = self.generate_maze_map()
        xshift = 0.33
        yshift = 0.33
        cmap = plt.cm.get_cmap('gray')
        norm = plt.Normalize(maze_map.min(), maze_map.max())
        img = cmap(norm(maze_map))

        if self.startNode is not None:
            x, y = self.node2xy(self.startNode)
            img[y, x, :3] = 0, 0, 1
            # plt.text(x - xshift, y - yshift, str(self.startNode), fontweight='bold') # always print startNode

        if self.sinkerNode is not None:
            x, y = self.node2xy(self.sinkerNode)
            img[y, x, :3] = 1, 0, 0
            # plt.text(x - xshift, y - yshift, str(self.sinkerNode), fontweight='bold') # always print sinkerNode

        if show:
            if show_nodes:
                for n in range(self.height * self.width):
                    x, y = self.node2xy(n)
                    plt.text(x - xshift, y - yshift, str(n), fontweight='bold')

            if show_links:
                for n in range(1, (self.height - 1) * self.width + (self.width - 1) * self.height + 1):
                    x, y = self.link2xy(n)
                    plt.text(x, y - yshift, str(n), style='italic', color='red')

            if show_ticks:
                plt.xticks(np.arange(0, img.shape[1], step=4),
                           np.arange(0, (img.shape[1] - 1) / 2, step=2, dtype='int'))
                plt.yticks(np.arange(0, img.shape[0], step=4),
                           np.arange(0, (img.shape[0] - 1) / 2, step=2, dtype='int'))
            else:
                plt.xticks([])
                plt.yticks([])

            ax = plt.imshow(img, origin='lower')
            plt.show()

        else:
            ax = None

        return img, ax

    def set_link(self, link=None, value=None):
        """Set the value of a link in the maze.

        It updates the adjacency matrix in the maze changing the value associated with a link.
        Links are indexed  from 1 to self.total_links = (self.height-1)*self.width + (self.width-1)*self.height.

        Parameters
        ----------
        link : int
            index of the link to change. Its value is between 1 and self.total_links = (self.height-1)*self.width +
            (self.width-1)*self.height. Other values leave the maze unchanged.
        value : numpy.float64
            value to set to the link in the adjacency matrix.

        Returns
        -------
        numpy.float64
            returns the value set to the link
        """
        assert 1 <= link <= self.total_links
        if 1 <= link <= self.vertical_links:
            row = (link - 1) // (self.height - 1) * self.height + (link - 1) % (
                    self.height - 1)
            col = row + 1

        elif self.vertical_links < link <= self.total_links:
            row = link - (self.height - 1) * self.width - 1
            col = row + self.height

        self.adjacency[row, col] = value
        self.adjacency[col, row] = value

        return value

    def get_link(self, link=None):
        """Get the value of a link in the maze.

        Parameters
        ----------
        link : int
            index of the link to change. Its value is between 1 and self.total_links = (self.height-1)*self.width +
            (self.width-1)*self.height. Other values leave the maze unchanged.

        Returns
        -------
        numpy.float64
            returns the value of the link
        """
        assert 1 <= link <= self.total_links
        if 1 <= link <= self.vertical_links:
            row = (link - 1) // (self.height - 1) * self.height + (link - 1) % (
                    self.height - 1)
            col = row + 1

        elif self.vertical_links < link <= self.total_links:
            row = link - (self.height - 1) * self.width - 1
            col = row + self.height

        return self.adjacency[row, col]

    def reverse_link(self, link=None):
        """Reverse the value of a link in the maze, from 0 to 1 and vice-versa.

        It updates the adjacency matrix in the maze changing the value associated with a link.
        Links are indexed  from 1 to self.total_links = (self.height-1)*self.width + (self.width-1)*self.height.

        Parameters
        ----------
        link : int
            index of the link to change. Its value is between 1 and self.total_links = (self.height-1)*self.width +
            (self.width-1)*self.height. Other values leave the maze unchanged.

        Returns
        -------
        numpy.float64
            returns the value set to the link
        """
        value = self.get_link(link)
        if value == 0:
            self.set_link(link, value=np.float64(1))
            return np.float64(1)
        elif value > 0:
            self.set_link(link, value=np.float64(0))
            return np.float64(0)

    def xy2node(self, x, y):
        """Convert the coordinates x,y of the maze map to the index of the node.

        Given a matrix representing a maze with size width and height, it converts the coordinates (x,y) to the index
        corresponding to a node. The nodes have even coordinates both in x and y, and are ordered along the columns,
        meaning that the 0-th node has coordinate (0,0), the 1-st has (0,2), ... the height-th has (2,0) and so on up to
        the width*height-1 th node with coordinates (2*width, 2*height).

        Parameters
        ----------
        x : int, even
            x coordinate of the node in the maze
        y : int, even
            y coordinate of the node in the maze

        Returns
        -------
        int
            Index representing the node, or numpy.nan if the parameters are invalid
        """
        assert 0 <= x <= 2 * self.width + 1
        assert 0 <= y <= 2 * self.height + 1
        if x % 2 == 0 and y % 2 == 0:
            n = int((x // 2) * self.height + y // 2)
        else:
            n = np.nan

        return n

    def node2xy(self, n):
        """Convert the index of the node into its coordinates x,y.

        Given a matrix representing a maze with size width and height, it converts the coordinates (x,y) to the index
        corresponding to a node. The nodes have even coordinates both in x and y, and are ordered along the columns,
        meaning that the 0-th node has coordinate (0,0), the 1-st has (0,2), ... the height-th has (2,0) and so on up to
        the width*height-1 th node with coordinates (2*width, 2*height).

        Parameters
        ----------
        n : int, from 0 to width*height-1
            index of the node

        Returns
        -------
        (int, int)
            Index representing the node
        """
        return 2 * (n // self.height), 2 * (n % self.height)

    def xy2link(self, x, y):
        """Convert the coordinates x,y of the maze map to the index of the link between two nodes.

        Given a matrix representing a maze with size width and height, it converts the coordinates (x,y) to the index
        corresponding to a link. The link are ordered with an index that runs first all the vertical links along the
        columns, then all the horizontal links, also along the columns.

        The vertical links are numbered from 1 to self.vertical_links, the horizontal links are numbered from
        self.vertical_links+1 to self.total_links. The 1-st link has coordinates (0,1), the 2-th has
        (0,3), ... the height-th has (2,1), ... the (height-1)*width+1 th (which is the first horizontal) has (1,0),
        then (1,2) and so on. Note that for a valid coordinate, x or y must be even.

        Parameters
        ----------
        x : int
            x coordinate of the link in the maze
        y : int
            y coordinate of the link in the maze

        Returns
        -------
        int
            Index representing the link, or numpy.nan if the parameters are invalid
        """
        if 0 <= x <= (self.width - 1) * 2 and x % 2 == 0:
            # vertical link
            n = int((x // 2) * (self.height - 1) + (y - 1) // 2 + 1)

        elif 0 <= y <= (self.height - 1) * 2 and y % 2 == 0:
            # horizontal link
            n = (x - 1) // 2 * self.height + y // 2 + 1
            n = int(n + (self.height - 1) * self.width)

        else:  # invalid parameters
            n = np.nan

        return n

    def link2xy(self, n):
        """Convert the index of the link between two nodes into its coordinates x,y in the maze map.

        Given a matrix representing a maze with size width and height, it converts the coordinates (x,y) to the index
        corresponding to a link. The link are ordered with an index that runs first all the vertical links along the
        columns, then all the horizontal links, also along the columns.

        The vertical links are numbered from 1 to self.vertical_links, the horizontal links are numbered from
        self.vertical_links+1 to self.total_links. The 1-st link has coordinates (0,1), the 2-th has
        (0,3), ... the height-th has (2,1), ... the (height-1)*width+1 th (which is the first horizontal) has (1,0),
        then (1,2) and so on. Note that for a valid coordinate, x or y must be even.

        Parameters
        ----------
        n : int
            index representing the link

        Returns
        -------
        (x,y)
            x and y coordinates of the link, or numpy.nan if the parameters are invalid
        """
        if 1 <= n <= self.vertical_links:  # vertical link
            x, y = 2 * ((n - 1) // (self.height - 1)), 2 * ((n - 1) % (self.height - 1)) + 1

        elif self.vertical_links < n <= self.total_links:  # horizontal link
            n = n - self.vertical_links
            x, y = 2 * ((n - 1) // self.height) + 1, 2 * ((n - 1) % self.height)

        else:  # invalid parameters
            x, y = np.nan, np.nan

        return x, y

    def save(self, filename=None):
        """Save a maze.

        Parameters
        ----------
        filename : str (default None)
            name of the file where to save the maze. If None, use current date and time.

        Returns
        -------
        str
            Index representing the link, or numpy.nan if the parameters are invalid
        """
        if filename is None:
            filename = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '_maze'
        if not filename.endswith('.pkl'):
            filename += '.pkl'

        with open(filename, 'wb') as f:
            pickle.dump(self.__dict__, f)

        return filename

    def load(self, filename):
        """Load a maze from a file.

         Parameters
        ----------
        filename : str (default None)
            name of the file where to save the maze. If None, use current date and time.
        """

        assert filename is not None, "filename parameter in Maze.load() is None"

        if not filename.endswith('.pkl'):
            filename += '.pkl'

        with open(filename, 'rb') as f:
            tmp_dict = pickle.load(f)

        self.__dict__.update(tmp_dict)

        return self


if __name__ == "__main__":
    print('maze_tools has started')
    # maze_tools test
    myMaze = Maze(maze_size=(20, 10))
    myMaze.plot_maze(show_nodes=True, show_links=False)
    myMaze.save('saving_test')
    # myMaze = None
    myMaze2 = Maze().load('saving_test')
    myMaze2.plot_maze(show_nodes=True, show_links=False)
    # comparison test between myMaze and myMaze2
    np.all([np.all(myMaze.__dict__[x] == myMaze2.__dict__[x]) for x in myMaze.__dict__.keys()])
