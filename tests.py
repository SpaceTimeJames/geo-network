from network.utils import network_point_coverage

__author__ = 'gabriel'
from network import TEST_DATA_FILE
from network.itn import read_gml, ITNStreetNet
from network.streetnet import NetPath, NetPoint, Edge, GridEdgeIndex
from data import models
import os
import unittest
import settings
import numpy as np
from matplotlib import pyplot as plt
from network import utils
from validation import hotspot, roc
import networkx as nx
from shapely.geometry import LineString


def load_test_network():
    # load some toy network data
    test_data = read_gml(TEST_DATA_FILE)
    return ITNStreetNet.from_data_structure(test_data)


def toy_network():
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
    ls = LineString([
        (0, 0),
        (2.5, 1),
        (5, 0)
    ])
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



if __name__ == "__main__":
    b_plot = False

    # mini test dataset

    # test dataset is in a directory in the same path as this module called 'test_data'
    this_dir = os.path.dirname(os.path.realpath(__file__))
    IN_FILE = os.path.join(this_dir, 'test_data', 'mastermap-itn_417209_0_brixton_sample.gml')
    test_data = read_gml(IN_FILE)
    itn_net = ITNStreetNet.from_data_structure(test_data)

    # buffered Camden dataset from raw data
    # test dataset is in a directory in the data directory called 'network_data'
    # this_dir = os.path.join(settings.DATA_DIR, 'network_data')
    # IN_FILE = os.path.join(this_dir, 'mastermap-itn_544003_0_camden_buff2000.gml')
    # test_data = read_gml(IN_FILE)
    # itn_net = ITNStreetNet.from_data_structure(test_data)

    # buffered Camden dataset from pickle
    # this_dir = os.path.dirname(os.path.realpath(__file__))
    # IN_FILE = os.path.join(this_dir, 'test_data', 'mastermap-itn_544003_0_camden_buff2000.pickle')
    # itn_net = ITNStreetNet.from_pickle(IN_FILE)

    # get the spatial extent of the network

    xmin, ymin, xmax, ymax = itn_net.extent

    # lay down some random points within that box
    num_pts = 100

    x_pts = np.random.rand(num_pts) * (xmax - xmin) + xmin
    y_pts = np.random.rand(num_pts) * (ymax - ymin) + ymin

    # now we want to snap them all to the network...
    # method A: do it in two steps...

    # A1: push them into a single data array for easier operation
    xy = models.DataArray.from_args(x_pts, y_pts)
    # A2: use the class method from_cartesian,
    net_point_array_a = models.NetworkData.from_cartesian(itn_net, xy, grid_size=50)  # grid_size defaults to 50

    # method B: do it manually, just to check
    # also going to take this opportunity to test a minor problem with closest_edges_euclidean

    grid_edge_index = itn_net.build_grid_edge_index(50)
    net_points = []
    snap_dists = []
    fail_idx = []
    for i, (x, y) in enumerate(zip(x_pts, y_pts)):
        tmp = itn_net.closest_edges_euclidean(x, y, grid_edge_index=grid_edge_index)
        if tmp[0] is None:
            # some of those calls fail when the grid_size is too small (e.g. 50 is actually too small)
            # the fall back should probably be a method that does not depend on the grid, which is what
            # closest_segments_euclidean_brute_force is designed to do
            # this method is MUCH slower but always finds an edge
            tmp = itn_net.closest_edges_euclidean_brute_force(x, y)
            fail_idx.append(i)
        net_points.append(tmp[0])
        snap_dists.append(tmp[1])

    net_point_array_b = models.NetworkData(net_points)

    # check these are the same

    print net_point_array_a == net_point_array_b  # this is just doing a point-by-point equality check behind the scenes

    # find the cartesian_coords after snapping
    xy_post_snap = net_point_array_a.to_cartesian()

    # plot showing the snapping operation

    # this separates the data arrays back into their constituent dims
    x_pre, y_pre = xy.separate
    x_post, y_post = xy_post_snap.separate

    if b_plot:

        fig = plt.figure()
        ax = fig.add_subplot(111)
        itn_net.plot_network(ax=ax, edge_width=7, edge_inner_col='w')
        ax.plot(x_pre, y_pre, 'ro')
        ax.plot(x_post, y_post, 'bo')
        [ax.plot([x_pre[i], x_post[i]], [y_pre[i], y_post[i]], 'k-') for i in range(xy.ndata)]

        # highlight failed points (where closest_edges_euclidean didn't find any snapped point) in black circles
        [ax.plot(x_pre[i], y_pre[i], marker='o', markersize=20, c='k', fillstyle='none') for i in fail_idx]

    # glue the network point array together with a time dimension - just take time at uniform intervals on [0, 1]
    st_net_point_array = models.NetworkSpaceTimeData(
        zip(np.linspace(0, 1, num_pts), net_points)
    )

    # compute linkages at a max delta t and delta d
    # i, j = network_linkages(st_net_point_array, max_t=1.0, max_d=5000.)

    # excise data with time cutoff (validation machinery does this for you normally)
    training_data = st_net_point_array.getrows(np.where(st_net_point_array.time <= 0.6)[0])
    training_t = training_data.toarray(0)
    training_xy = training_data.space.to_cartesian()
    testing_data = st_net_point_array.getrows(np.where(st_net_point_array.time > 0.6)[0])

    # create instance of Bowers ProMap network kernel
    h = hotspot.STNetworkBowers(1000, 2)

    # bind it to data
    h.train(training_data)

    # instantiate Roc
    r = roc.NetworkRocSegments(data=testing_data.space, graph=itn_net)
    r.set_sample_units(None)
    prediction_points_net = r.sample_points
    prediction_points_xy = prediction_points_net.to_cartesian()

    z = h.predict(0.6, prediction_points_net)
    r.set_prediction(z)

    if b_plot:
        # show the predicted values, training data and sampling points
        r.plot()
        plt.scatter(training_xy.toarray(0), training_xy.toarray(1), c=training_t, cmap='jet', s=40)
        plt.plot(prediction_points_xy.toarray(0), prediction_points_xy.toarray(1), 'kx', markersize=20)
        plt.colorbar()

    # repeat for a more accurate Roc class that uses multiple readings per segment
    if False:  # disable for now
        r2 = roc.NetworkRocSegmentsMean(data=testing_data.space, graph=itn_net)
        r2.set_sample_units(None, 10)
        prediction_points_net2 = r2.sample_points
        prediction_points_xy2 = prediction_points_net2.to_cartesian()

        z2 = h.predict(0.6, prediction_points_net2)
        r2.set_prediction(z2)

        if b_plot:
            # show the predicted values, training data and sampling points
            r2.plot()
            plt.scatter(training_xy.toarray(0), training_xy.toarray(1), c=training_t, cmap='jet', s=40)
            plt.plot(prediction_points_xy2.toarray(0), prediction_points_xy2.toarray(1), 'kx', markersize=20)
            plt.colorbar()

    # get a roughly even coverage of points across the network
    net_points, edge_count = network_point_coverage(itn_net, dx=10)
    xy_points = net_points.to_cartesian()
    c_edge_count = np.cumsum(edge_count)

    # make a 'prediction' for time 1.1
    # st_net_prediction_array = models.DataArray(
    #     np.ones(net_points.ndata) * 1.1
    # ).adddim(net_points, type=models.NetworkSpaceTimeData)

    # z = h.predict(st_net_prediction_array)

    if b_plot:
        # get colour limits - otherwise single large values dominate the plot
        fmax = 0.7
        vmax = sorted(z)[int(len(z) * fmax)]

        plt.figure()
        itn_net.plot_network(edge_width=8, edge_inner_col='w')
        plt.scatter(xy_points[:, 0], xy_points[:,1], c=z, cmap='Reds', vmax=vmax, s=50, edgecolor='none', zorder=3)
        # j = 0
        # for i in range(len(itn_net.edges())):
        #     n = c_edge_count[i]
        #     x = xy_points[j:n, 0]
        #     y = xy_points[j:n, 1]
        #     val = z[j:n]
        #     plotting.colorline(x, y, val, linewidth=8)
        #     j = n

    from network import utils
    # n_iter = 30
    g = utils.network_walker(itn_net, verbose=False, repeat_edges=False)
    # res = [g.next() for i in range(n_iter)]
    res = list(g)

    import matplotlib.lines as mlines
    import matplotlib.patches as mpatches

    def add_arrow_to_line2D(
        line,
        axes=None,
        arrowstyle='-|>',
        arrowsize=1):
        """
        Add arrows to a matplotlib.lines.Line2D at the midpoint.

        Parameters:
        -----------
        axes:
        line: list of 1 Line2D obbject as returned by plot command
        arrowstyle: style of the arrow
        arrowsize: size of the arrow
        transform: a matplotlib transform instance, default to data coordinates

        Returns:
        --------
        arrows: list of arrows
        """
        axes = axes or plt.gca()
        if (not(isinstance(line, list)) or not(isinstance(line[0],
                                               mlines.Line2D))):
            raise ValueError("expected a matplotlib.lines.Line2D object")
        x, y = line[0].get_xdata(), line[0].get_ydata()

        arrow_kw = dict(arrowstyle=arrowstyle, mutation_scale=10 * arrowsize)

        color = line[0].get_color()
        use_multicolor_lines = isinstance(color, np.ndarray)
        if use_multicolor_lines:
            raise NotImplementedError("multicolor lines not supported")
        else:
            arrow_kw['color'] = color

        linewidth = line[0].get_linewidth()
        if isinstance(linewidth, np.ndarray):
            raise NotImplementedError("multiwidth lines not supported")
        else:
            arrow_kw['linewidth'] = linewidth

        sc = np.concatenate(([0], np.cumsum(np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2))))
        x0 = np.interp(0.45 * sc[-1], sc, x)
        y0 = np.interp(0.45 * sc[-1], sc, y)
        x1 = np.interp(0.55 * sc[-1], sc, x)
        y1 = np.interp(0.55 * sc[-1], sc, y)
        # s = np.cumsum(np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2))
        # n = np.searchsorted(s, s[-1] * loc)
        # arrow_tail = (x[n], y[n])
        # arrow_head = (np.mean(x[n:n + 2]), np.mean(y[n:n + 2]))
        arrow_tail = (x0, y0)
        arrow_head = (x1, y1)
        p = mpatches.FancyArrowPatch(
            arrow_tail, arrow_head, transform=axes.transData,
            **arrow_kw)
        axes.add_patch(p)

        return p

    if b_plot:
        fig = plt.figure(figsize=(16, 12))
        itn_net.plot_network()

        for i in range(len(res)):
            node_loc = itn_net.g.node[res[i][0][-1]]['loc']
            h = plt.plot(node_loc[0], node_loc[1], 'ko', markersize=10)[0]
            edge_x, edge_y = res[i][2].linestring.xy
            # which way are we walking?
            if res[i][2].orientation_pos == res[i][0][-1]:
                # need to reverse the linestring
                edge_x = edge_x[::-1]
                edge_y = edge_y[::-1]
            line = plt.plot(edge_x, edge_y, 'k-')
            add_arrow_to_line2D(line, arrowsize=2)
            fig.savefig('/home/gabriel/tmp/%02d.png' % i)
            h.remove()
            if i + 1 == len(res):
                # final image - save it 25 more times to have a nice lead out
                for j in range(25):
                    fig.savefig('/home/gabriel/tmp/%02d.png' % (j + i))

    # run to stitch images together:
    # avconv -r 10 -crf 20 -i "%02d.png" -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -c:v libx264 -pix_fmt yuv420p output.mp4

    # network KDE stuff
    from kde import models as kde_models, kernels

    prediction_points_tnet = hotspot.generate_st_prediction_dataarray(0.6,
                                                                      prediction_points_net,
                                                                      dtype=models.NetworkSpaceTimeData)

    a = kde_models.NetworkFixedBandwidthKde(training_data, bandwidths=[5., 50.], parallel=False)
    res = a.pdf(prediction_points_tnet)

    if b_plot:
        itn_net.plot_network()
        plt.scatter(*training_data.space.to_cartesian().separate, c='r', s=80)
        plt.scatter(*prediction_points_net.to_cartesian().separate, c=res/res.max(), s=40)


