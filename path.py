import math
import numpy as np
from shapely.geometry import LineString


class NetPath(object):

    ## TODO: consider adding a linestring property - it would be a nice way to verify things

    def __init__(self, net, start, end, nodes, distance, edges=None, split=None):
        """
        Object encoding a route between two points on the network, either NetPoint or existing nodes
        :param start: Start Node/NetPoint
        :param end: Terminal Node/NetPoint
        :param nodes: List of the nodes traversed
        :param distance: Either an array of individual edge distances, or a float containing the total
        :param edges: Optional array of Edge objects traversed
        :param splits: Optional, either an array of node splits or a float containing the product
        :return:
        """
        self.start = start
        self.end = end
        self.nodes = nodes

        self._total_distance = None
        self.distances = None
        if hasattr(distance, '__iter__'):
            self.distances = distance
            self.distance_total = sum(distance)
        else:
            self.distance_total = distance
        self.edges = edges
        self._splits = None
        self._split_total = None
        if split is not None:
            if hasattr(split, '__iter__'):
                self._splits = split
                self._split_total = np.prod(split)
            else:
                self._split_total = split

        if self.edges is not None and self.distances is not None and (len(self.distances) != len(edges)):
            raise AttributeError('Path mismatch: distance list wrong length')

        if self.edges is not None and len(nodes) != len(edges) - 1:
            raise AttributeError('Path mismatch: node list wrong length')

        # if self.start.graph is not self.end.graph:
        #     raise AttributeError('Path mismatch: nodes are defined on different graphs')

        self.graph = net

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    @property
    def splits(self):
        if self._splits is None:
            self._splits = [max(self.graph.g.degree(t) - 1, 1) for t in self.nodes]
        return self._splits

    @property
    def splits_total(self):
        if self._split_total is None:
            self._split_total = np.prod(self.splits)
        return self._split_total

    @property
    def length(self):
        return self.distance_total
