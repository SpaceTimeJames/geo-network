import numpy as np
from data.models import NetworkData
from point import NetPoint
from network import itn
from network.utils import network_walker_fixed_distance
from networkx import MultiGraph
from shapely.geometry import LineString


def weighted_random_selection(weights, n=1, prng=None):
    """
    Select the *indices* of n points at random, weighted by the supplied weights matrix
    :param weights:
    :param n:
    :param prng: Optionally supply a np.random.Random() instance if control is required
    :return: Iterable of n indices, referencing the weights array, or scalar index if n == 1
    """
    prng = prng or np.random.RandomState()
    totals = np.cumsum(weights)
    throws = prng.rand(n) * totals[-1]
    res = np.searchsorted(totals, throws)
    if n == 1:
        return res[0]
    else:
        return res


def create_grid_network(domain_extents,
                        row_spacing,
                        col_spacing):
    """
    Create a Manhattan network with horizontal / vertical edges.
    :param domain_extents: (xmin, ymin, xmax, ymax) of the domain
    :param row_spacing: Distance between horizontal edges
    :param col_spacing: Distance between vertical edges
    :return: Streetnet object
    """
    xmin, ymin, xmax, ymax = domain_extents
    # compute edge coords
    y = np.arange(ymin + row_spacing / 2., ymax, row_spacing)
    x = np.arange(xmin + col_spacing / 2., xmax, col_spacing)
    g = MultiGraph()
    letters = []
    aint = ord('a')
    for i in range(26):
        for j in range(26):
            for k in range(26):
                letters.append(chr(aint + i) + chr(aint + j) + chr(aint + k))

    def add_edge(x0, y0, x1, y1):
        if x0 < 0 or y0 < 0 or x1 >= len(x) or y1 >= len(y):
            # no link needed
            return
        k0 = x0 * x.size + y0
        k1 = x1 * x.size + y1
        idx_x0 = letters[k0]
        idx_x1 = letters[k1]
        label0 = idx_x0 + str(y0)
        label1 = idx_x1 + str(y1)
        ls = LineString([
            (x[x0], y[y0]),
            (x[x1], y[y1]),
        ])
        atts = {
            'fid': "%s-%s" % (label0, label1),
            'linestring': ls,
            'length': ls.length,
            'orientation_neg': label0,
            'orientation_pos': label1
        }
        g.add_edge(label0, label1, key=atts['fid'], attr_dict=atts)
        g.node[label0]['loc'] = (x[x0], y[y0])
        g.node[label1]['loc'] = (x[x1], y[y1])

    for i in range(x.size):
        for j in range(y.size):
            add_edge(i, j-1, i, j)
            add_edge(i-1, j, i, j)
            add_edge(i, j, i, j+1)
            add_edge(i, j, i+1, j)

    return itn.ITNStreetNet.from_multigraph(g)


def uniform_random_points_on_net(net, n=1):
    """
    Draw n NetPoints at random that lie on the supplied network
    :param net:
    :param n: Number of points to draw
    :return: NetworkData array if n>1, else NetPoint
    """
    all_edges = net.edges()

    # segment lengths
    ls = np.array([e.length for e in all_edges])

    # random edge draw weighted by segment length
    if n == 1:
        selected_edges = [all_edges[weighted_random_selection(ls, n=n)]]
    else:
        ind = weighted_random_selection(ls, n=n)
        selected_edges = [all_edges[i] for i in ind]

    # random location along each edge
    frac_along = np.random.rand(n)
    res = []
    for e, fa in zip(selected_edges, frac_along):
        dist_along = {
            e.orientation_neg: e.length * fa,
            e.orientation_pos: e.length * (1 - fa),
        }
        the_pt = NetPoint(
            net,
            e,
            dist_along
        )
        res.append(the_pt)

    if n == 1:
        return res[0]
    else:
        return NetworkData(res)


def random_walk_normal(net_pt, sigma=1.):
    """
    Starting from net_pt, take a random walk along the network with distance drawn from a normal distribution with
    stdev sigma.
    :param net_pt:
    :param sigma:
    :return: NetPoint instance.
    """
    dist = np.abs(np.random.randn() * sigma)
    pts, _ = network_walker_fixed_distance(net_pt.graph, net_pt, dist)
    return pts[np.random.randint(len(pts))]
