import numpy as np
from data.models import NetworkData
from network.streetnet import NetPoint
from network.utils import network_walker_fixed_distance
from stats import random


def uniform_random_points_on_net(net, n=1):
    """
    Draw n NetPoints at random that lie on the supplied network
    :param net:
    :param n:
    :return: NetworkData array if n>1, else NetPoint
    """
    all_edges = net.edges()

    # segment lengths
    ls = np.array([e.length for e in all_edges])

    # random edge draw weighted by segment length
    if n == 1:
        selected_edges = [all_edges[random.weighted_random_selection(ls, n=n)]]
    else:
        ind = random.weighted_random_selection(ls, n=n)
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
