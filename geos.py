__author__ = 'gabriel'
import networkx as nx
import math


class NetworkPoint(dict):

    def __init__(self, graph, **dist_along):
        # copy a POINTER to graph: if graph object changes then this will also change
        super(NetworkPoint, self).__init__(**dist_along)
        self.g = graph
        if self.__len__() != 2:
            raise AttributeError("Initialiser requires 2 kwargs")
        for v in self.values():
            if not (isinstance(v, int) or isinstance(v, float)):
                raise TypeError("Supplied distance values are not numeric")

    def network_distance(self, other):
        if not other.g is self.g:
            raise TypeError("Cannot compare network points with different underlying graphs")

        # also deal with trivial case of two points on same edge
        num_matches = 0

        distances=[]

        for k1, da1 in self.items():
            for k2, da2 in other.items():
                if k1 == k2:
                    num_matches += 1
                #Get the network distance between terminal nodes
                network_distance = nx.dijkstra_path_length(self.g, k1, k2, 'length')

                #Add the extra distance at each end of the route
                total_distance = network_distance + da1 + da2

                #Add to the list
                distances.append(total_distance)

        if num_matches > 1:
            # same edge - reuse k1 and k2
            return math.fabs(self[k1] - other[k1])

        return min(distances)