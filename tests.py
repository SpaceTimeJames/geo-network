__author__ = 'gabriel'
from __init__ import TEST_DATA_FILE
import utils
from itn import read_gml, ITNStreetNet
from streetnet import NetPath, NetPoint, Edge, GridEdgeIndex
from data import models
import unittest
import numpy as np

import networkx as nx
from shapely.geometry import LineString


def load_test_network():
    # load some toy network data
    test_data = read_gml(TEST_DATA_FILE)
    return ITNStreetNet.from_data_structure(test_data)


def toy_network(loop=False):
    g = nx.MultiGraph()
    node_coords = {
        'a': (0, 0),
        'b': (5, 0),
        'c': (5, 2),
        'd': (5, 3),
        'e': (6, 2),
        'f': (7, 0),
        'g': (7, -2),
        'h': (5, -2),
        'i': (5, -3),
        'j': (4, -2),
        'k': (0, -2),
        'l': (-1, -2),
        'm': (-2 ** .5 / 2., -2 ** .5 / 2. - 2),
        'n': (0, -3),
        'o': (1, -2),
        'p': (0, 2),
    }
    edges = [
        ('a', 'b'),
        ('a', 'k'),
        ('k', 'l'),
        ('k', 'm'),
        ('k', 'n'),
        ('k', 'o'),
        ('b', 'c'),
        ('b', 'h'),
        ('c', 'd'),
        ('c', 'e'),
        ('b', 'f'),
        ('f', 'g'),
        ('g', 'h'),
        ('b', 'h'),
        ('h', 'j'),
        ('h', 'i'),
        ('a', 'p')
    ]
    def attr_factory(start, end):
        xy0 = node_coords[start]
        xy1 = node_coords[end]
        ls = LineString([xy0, xy1])
        attr_dict = {
            'linestring': ls,
            'length': ls.length,
            'fid': start + end + '1',
            'orientation_neg': start,
            'orientation_pos': end
        }
        return attr_dict

    for i0, i1 in edges:
        attr = attr_factory(i0, i1)
        g.add_edge(i0, i1, key=attr['fid'], attr_dict=attr)

    # add 2 more multilines between a and b
    attr = attr_factory('a', 'b')
    th = np.linspace(0, np.pi, 50)[::-1]
    x = 2.5 * (np.cos(th) + 1)
    y = np.sin(th)
    ls = LineString(zip(x, y))

    attr['fid'] = 'ab2'
    attr['linestring'] = ls
    attr['length'] = ls.length
    g.add_edge('a', 'b', key=attr['fid'], attr_dict=attr)
    ls = LineString([
        (0, 0),
        (2.5, -1),
        (5, 0)
    ])
    attr['fid'] = 'ab3'
    attr['linestring'] = ls
    g.add_edge('a', 'b', key=attr['fid'], attr_dict=attr)

    if loop:
        # add cycle at p
        attr = attr_factory('p', 'p')
        th = np.linspace(-np.pi / 2., 3 * np.pi / 2., 50)
        x = np.cos(th)
        y = np.sin(th) + node_coords['p'][1] + 1
        ls = LineString(zip(x, y))
        attr['linestring'] = ls
        attr['length'] = ls.length
        g.add_edge('p', 'p', key=attr['fid'], attr_dict=attr)


    # add node coords
    for k, v in node_coords.items():
        g.node[k]['loc'] = v

    net = ITNStreetNet.from_multigraph(g)
    return net


class TestNetworkData(unittest.TestCase):

    def setUp(self):
        # this_dir = os.path.dirname(os.path.realpath(__file__))
        # IN_FILE = os.path.join(this_dir, 'test_data', 'mastermap-itn_417209_0_brixton_sample.gml')

        self.test_data = read_gml(TEST_DATA_FILE)

        self.itn_net = ITNStreetNet.from_data_structure(self.test_data)

    def test_grid_index(self):
        xmin, ymin, xmax, ymax =  self.itn_net.extent
        grid_edge_index = self.itn_net.build_grid_edge_index(50)
        x_grid_expct = np.arange(xmin, xmax, 50)
        self.assertTrue(np.all(grid_edge_index.x_grid == x_grid_expct))

    def test_extent(self):
        expected_extent = (530960.0, 174740.0, 531856.023, 175436.0)
        for eo, ee in zip(expected_extent, self.itn_net.extent):
            self.assertAlmostEqual(eo, ee)

    def test_net_point(self):
        #Four test points - 1 and 3 on same segment, 2 on neighbouring segment, 4 long way away.
        #5 and 6 are created so that there are 2 paths of almost-equal length between them - they
        #lie on opposite sides of a 'square'
        x_pts = (
            531190,
            531149,
            531210,
            531198,
            531090
        )
        y_pts = (
            175214,
            175185,
            175214,
            174962,
            175180
        )
        xmin, ymin, xmax, ymax = self.itn_net.extent
        grid_edge_index = self.itn_net.build_grid_edge_index(50)
        net_points = []
        snap_dists = []
        for x, y in zip(x_pts, y_pts):
            tmp = self.itn_net.closest_edges_euclidean(x, y, grid_edge_index=grid_edge_index)
            net_points.append(tmp[0])
            snap_dists.append(tmp[0])

        # test net point subtraction
        self.assertIsInstance(net_points[1] - net_points[0], NetPath)
        self.assertAlmostEqual((net_points[1] - net_points[0]).length, (net_points[0] - net_points[1]).length)
        for i in range(len(net_points)):
            self.assertEqual((net_points[i] - net_points[i]).length, 0.)

        net_point_array = models.NetworkData(net_points)

        self.assertFalse(np.any(net_point_array.distance(net_point_array).data.sum()))

    def test_snapping_brute_force(self):
        # lay down some known points
        coords = [
            (531022.868, 175118.877),
            (531108.054, 175341.141),
            (531600.117, 175243.572),
            (531550, 174740),
        ]
        # the edges they should correspond to
        edge_params = [
            {'orientation_neg': 'osgb4000000029961720_0',
             'orientation_pos': 'osgb4000000029961721_0',
             'fid': 'osgb4000000030340202'},
            {'orientation_neg': 'osgb4000000029962839_0',
             'orientation_pos': 'osgb4000000029962853_0',
             'fid': 'osgb4000000030235941'},
            {'orientation_neg': 'osgb4000000030778079_0',
             'orientation_pos': 'osgb4000000030684375_0',
             'fid': 'osgb4000000030235965'},
            None,  # no edge within radius
        ]
        for c, e in zip(coords, edge_params):
            # snap point
            this_netpoint = NetPoint.from_cartesian(self.itn_net, *c, radius=50)
            # check edge equality
            if e:
                this_edge = Edge(self.itn_net, **e)
                self.assertEqual(this_netpoint.edge, this_edge)
            else:
                self.assertTrue(this_netpoint is None)

    def test_snapping_indexed(self):
        # lay down some known points
        coords = [
            (531022.868, 175118.877),
            (531108.054, 175341.141),
            (531600.117, 175243.572),
            (531550, 174740),
        ]
        # the edges they should correspond to
        edge_params = [
            {'orientation_neg': 'osgb4000000029961720_0',
             'orientation_pos': 'osgb4000000029961721_0',
             'fid': 'osgb4000000030340202'},
            {'orientation_neg': 'osgb4000000029962839_0',
             'orientation_pos': 'osgb4000000029962853_0',
             'fid': 'osgb4000000030235941'},
            {'orientation_neg': 'osgb4000000030778079_0',
             'orientation_pos': 'osgb4000000030684375_0',
             'fid': 'osgb4000000030235965'},
            None,  # no edge within radius
        ]
        gei = self.itn_net.build_grid_edge_index(50)
        # supply incompatible radius
        with self.assertRaises(AssertionError):
            this_netpoint = NetPoint.from_cartesian(self.itn_net, *coords[0], grid_edge_index=gei, radius=51)
        for c, e in zip(coords, edge_params):
            # snap point
            this_netpoint = NetPoint.from_cartesian(self.itn_net, *c, grid_edge_index=gei, radius=50)
            # check edge equality
            if e:
                this_edge = Edge(self.itn_net, **e)
                self.assertEqual(this_netpoint.edge, this_edge)
            else:
                self.assertTrue(this_netpoint is None)

        # retest last point without a radius
        e = {'orientation_neg': 'osgb4000000029961762_0',
             'orientation_pos': 'osgb4000000029961741_0',
             'fid': 'osgb4000000030145824'}
        c = coords[-1]
        this_netpoint = NetPoint.from_cartesian(self.itn_net, *c, grid_edge_index=gei)
        this_edge = Edge(self.itn_net, **e)
        self.assertEqual(this_netpoint.edge, this_edge)


class TestUtils(unittest.TestCase):
    def setUp(self):
        self.test_data = read_gml(TEST_DATA_FILE)
        self.itn_net = ITNStreetNet.from_data_structure(self.test_data)

    def test_network_edge_walker(self):
        g = utils.network_walker(self.itn_net, repeat_edges=False, verbose=False)
        res = list(g)
        # if repeat_edges == False. every edge should be covered exactly once
        self.assertEqual(len(res), len(self.itn_net.edges()))
        # since no start node was supplied, walker should have started at node 0
        self.assertEqual(res[0][0].nodes, [self.itn_net.nodes()[0]])
        # restart walk at a different node
        g = utils.network_walker(self.itn_net, repeat_edges=False, verbose=False, source_node=self.itn_net.nodes()[-1])
        res2 = list(g)
        self.assertEqual(len(res2), len(self.itn_net.edges()))

        # now run it again using the class
        obj = utils.NetworkWalker(self.itn_net,
                                  [],
                                  repeat_edges=False)
        g = obj.walker()
        res3 = list(g)
        self.assertListEqual(res, res3)

        # test caching
        start = self.itn_net.nodes()[0]
        self.assertTrue(start in obj.cached_walks)
        self.assertListEqual(res, obj.cached_walks[start])

        start = self.itn_net.nodes()[-1]
        g = obj.walker(start)
        res4 = list(g)
        self.assertListEqual(res2, res4)
        self.assertTrue(start in obj.cached_walks)
        self.assertListEqual(res2, obj.cached_walks[start])


    def test_fixed_distance_walk(self):
        net = toy_network()
        pt = NetPoint.from_cartesian(net, 2.5, 0)

