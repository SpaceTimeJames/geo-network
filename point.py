import numpy as np
import bisect as bs
from shapely.geometry import LineString


class NetPoint(object):

    def __init__(self, street_net, edge, node_dist):
        """
        :param street_net: A pointer to the network on which this point is defined
        :param edge: An Edge object
        :param node_dist: A dictionary containing the distance along this edge from both the positive and negative end
        The key gives the node ID, the value gives the distance from that end.
        """
        self.graph = street_net
        self.edge = edge
        # check node_dist
        assert len(node_dist) in (1, 2), "node_dist must have one or two entries"
        if len(node_dist) == 2:
            assert {edge.orientation_pos, edge.orientation_neg} == set(node_dist.keys()), \
                "node_dist keys do not match the edge nodes"
        else:
            # fill in other node_dist
            if node_dist.keys()[0] == edge.orientation_neg:
                node_dist[edge.orientation_pos] = self.edge.length - node_dist.values()[0]
            elif node_dist.keys()[0] == edge.orientation_pos:
                node_dist[edge.orientation_neg] = self.edge.length - node_dist.values()[0]
            else:
                raise AssertionError("The entry in node_dist does not match either of the edge nodes")
        self.node_dist = node_dist

    @classmethod
    def from_cartesian(cls, street_net, x, y, grid_edge_index=None, radius=None):
        """
        Convert from Cartesian coordinates to a NetPoint
        :param street_net: The governing network
        :param x:
        :param y:
        :param grid_edge_index: Optionally supply a pre-defined grid index, which allows for fast searching
        :param radius: Optional maximum search radius. If no edges are found within this radius, the point is not snapped.
        :return: NetPoint or None (if snapping fails)
        """
        ## TODO: update with new snapping method
        if grid_edge_index is not None:
            return street_net.closest_edges_euclidean(x, y,
                                                      grid_edge_index=grid_edge_index,
                                                      radius=radius)[0]
        else:
            res = street_net.closest_edges_euclidean_brute_force(x, y, radius=radius)
            if res is not None:
                return res[0]
            else:
                return

    @property
    def distance_positive(self):
        """ Distance from the POSITIVE node """
        return self.node_dist[self.edge.orientation_pos]

    @property
    def distance_negative(self):
        """ Distance from the NEGATIVE node """
        return self.node_dist[self.edge.orientation_neg]

    @property
    def cartesian_coords(self):
        ls = self.edge['linestring']
        pt = ls.interpolate(self.node_dist[self.edge.orientation_neg])
        return pt.x, pt.y

    def test_compatible(self, other):
        if not self.graph is other.graph:
            raise AttributeError("The two points are defined on different graphs")

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        # don't use test_compatible here because we want such an operation to return False, not raise an exception
        return (
            self.graph is other.graph and
            self.edge == other.edge and
            self.node_dist == other.node_dist
        )

    def __sub__(self, other):
        # NetPoint - NetPoint -> NetPath
        self.test_compatible(other)
        if self.graph.directed:
            return self.graph.path_directed(self, other)
        else:
            return self.graph.path_undirected(self, other)

    def distance(self, other, method=None):
        """
        NetPoint.distance(NetPoint) -> scalar distance
        :param method: optionally specify the algorithm used to compute distance.
        """
        if self.graph.directed:
            # TODO: update the path_directed method to accept length_only kwarg
            # return self.graph.path_directed(self, other, length_only=True, method=method)
            return (self - other).length
        else:
            return self.graph.path_undirected(self, other, length_only=True, method=method)


    def euclidean_distance(self, other):
        # compute the Euclidean distance between two NetPoints
        delta = np.array(self.cartesian_coords) - np.array(other.cartesian_coords)
        return sum(delta ** 2) ** 0.5

    def linestring(self, node):
        """ Partial edge linestring from/to the specified node. Direction is always neg to pos. """
        if node == self.edge.orientation_neg:
            return self.linestring_negative
        elif node == self.edge.orientation_pos:
            return self.linestring_positive
        else:
            raise AttributeError("Specified node is not one of the terminal edge nodes")

    @property
    def linestring_positive(self):
        """ Partial edge linestring from point to positive node """
        # get point coord using interp
        x, y = self.edge.linestring.xy
        d = np.concatenate(([0], np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2).cumsum()))
        i = bs.bisect_left(d, self.distance_negative)
        xp = np.interp(self.distance_negative, d, x)
        yp = np.interp(self.distance_negative, d, y)
        x = np.concatenate(([xp], x[i:]))
        y = np.concatenate(([yp], y[i:]))
        return LineString(zip(x, y))

    @property
    def linestring_negative(self):
        """ Partial edge linestring from negative node to point """
        # get point coord using interp
        x, y = self.edge.linestring.xy
        d = np.concatenate(([0], np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2).cumsum()))
        i = bs.bisect_left(d, self.distance_negative)
        xp = np.interp(self.distance_negative, d, x)
        yp = np.interp(self.distance_negative, d, y)
        x = np.concatenate((x[:i], [xp]))
        y = np.concatenate((y[:i], [yp]))
        return LineString(zip(x, y))

    @property
    def lineseg(self):
        """
        Line segment on which the point lies from negative node to positive node.
        """
        from streetnet import LineSeg
        x, y = self.edge.linestring.xy
        d = np.concatenate(([0], np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2).cumsum()))
        # when searching for the segment, we need to protect against points that lie exactly on the negative node
        # (these are still on the first segment)
        i = bs.bisect_left(d, self.distance_negative, lo=1)
        x = x[(i - 1):(i + 1)]
        y = y[(i - 1):(i + 1)]
        return LineSeg(
            node_neg_coords=(x[0], y[0]),
            node_pos_coords=(x[1], y[1]),
            edge=self.edge,
            street_net=self.graph
        )


class NetPointArray(object):

    def __init__(self, network_points, strict=True, copy=True):
        """
        Create an array of network points.
        :param network_points: iterable containing instances of NetPoint
        :param strict: If True (default) then check compatibility of NetPoint objects
        :param copy: If True (default) then copy the data when creating the array
        :return:
        """
        self.arr = np.array(network_points, copy=copy)
        if strict:
            for x in self.arr:
                if x.graph is not self.graph:
                    raise AttributeError("All network points must be defined on the same graph")

    def __repr__(self):
        return "{0}({1})".format(self.__class__.__name__, self.arr.__str__())

    def __iter__(self):
        return self.arr.__iter__()

    def __getitem__(self, item):
        return self.arr.__getitem__(item)

    def __sub__(self, other):
        return self.arr - other.arr

    @property
    def space(self):
        """
        Get the spatial component. In the base class, this is just the object itself. In derived classes, it is only
        one dimension of the array.
        """
        return self

    @space.setter
    def space(self, space):
        self.arr = np.array(space)

    @property
    def graph(self):
        if self.ndata:
            return self.arr[0].graph
        else:
            return None

    @classmethod
    def from_cartesian(cls, net, data, max_snap_distance=None, return_failure_idx=False):
        """
        Generate a NetworkData object for the (x, y) coordinates in data.
        :param net: The StreetNet object that will be used to snap network points.
        :param data: N x 2 numpy array or data that can be used to instantiate one.
        :param max_snap_distance: Optionally supply a maximum snapping distance.
        :param return_failure_idx: If True, output includes the index of points that did not snap. Otherwise, failed
        snaps are removed silently from the array.
        :return: NetPointArray object
        """
        data = np.array(data)
        if len(data.shape) != 2:
            raise AttributeError("Input data must be 2D")
        if data.shape[1] != 2:
            raise AttributeError("Input data must be 2D")

        net_points = []
        fail_idx = []
        for i, (x, y) in enumerate(data):
            t = NetPoint.from_cartesian(net, x, y, radius=max_snap_distance)
            if t is None:
                if return_failure_idx:
                    fail_idx.append(i)
            else:
                net_points.append(t)
        obj = cls(net_points)
        if return_failure_idx:
            return obj, fail_idx
        else:
            return obj

    def to_cartesian(self):
        """
        Convert all network points into Cartesian coordinates using linear interpolation of the edge LineStrings
        :return: CartesianData
        """
        return np.array([t.cartesian_coords for t in self.arr])

    @property
    def ndata(self):
        return len(self.arr)

    def distance(self, other, directed=False):
        # distance between self and other
        if not self.ndata == other.ndata:
            raise AttributeError("Lengths of the two data arrays are incompatible")
        return np.array([x.distance(y) for (x, y) in zip(self.space.arr, other.space.arr)])

    def euclidean_distance(self, other):
        """ Euclidean distance between the data """
        return np.array([x.euclidean_distance(y) for (x, y) in zip(self.space.arr, other.space.arr)])


class NetTimePointArray(object):

    def __init__(self, time_points, network_points, strict=True, copy=True):
        """
        Create an array of network-time points.
        :param time_points: iterable containing times
        :param network_points: iterable containing instances of NetPoint
        :param strict: If True (default) then check compatibility of NetPoint objects
        :param copy: If True (default) then copy the data when creating the array
        :return:
        """
        if isinstance(network_points, NetPointArray):
            self.s = network_points.arr
        else:
            self.s = np.array(network_points, copy=copy)
        self.t = np.array(time_points, copy=copy)
        if self.s.size != self.t.size:
            raise AttributeError("The net point and time arrays are not of equal sizes.")
        self.arr = np.vstack((self.t, self.s)).transpose()
        if strict:
            for x in self.space.arr:
                if x.graph is not self.graph:
                    raise AttributeError("All network points must be defined on the same graph")

    def __repr__(self):
        return "{0}({1})".format(self.__class__.__name__, self.arr.__str__())

    def __iter__(self):
        return self.arr.__iter__()

    def __getitem__(self, item):
        return self.arr.__getitem__(item)

    def __sub__(self, other):
        return self.arr - other.arr

    @property
    def graph(self):
        return self.space.graph

    @property
    def space(self):
        return NetPointArray(self.s, strict=False, copy=False)

    @space.setter
    def space(self, space):
        self.s = space

    @property
    def time(self):
        return self.t

    @time.setter
    def time(self, time):
        self.t = time

