__author__ = 'gabriel'
import numpy as np
from point_process.utils import linkages, linkage_func_separable
from data.models import CartesianSpaceTimeData
import logging
from streetnet import NetPoint
from collections import OrderedDict
import operator


#TODO: update this to use a function too, then implement that in hotspot.STNetworkBowers
def network_linkages(data_source_net,
                     linkage_fun,
                     data_source_txy=None,
                     data_target_net=None,
                     data_target_txy=None,
                     chunksize=2**18,
                     remove_coincident_pairs=False):
    """
    Compute the indices of datapoints that are within the following tolerances:
    interpoint distance less than max_d
    time difference greater than zero, less than max_t
    The sign convention is (target - source).
    This is almost identical to point_process.utils.linkages, with one addition: because network distance searches can
    be slow, we first test the Euclidean distance as a lower bound, then only compute net distances if that is within
    the tolerances.
    :param data_source_net: NetworkSpaceTimeData array of source data.  Must be sorted by time ascending.
    :param data_source_txy: Optional EuclideanSpaceTimeData array of source data.  Must be sorted by time ascending.
    If not supplied, compute from the network points.
    :param linkage_fun: Function that accepts two DataArrays (dt, dd) and returns an array of bool indicating whether
    the link with those distances is permitted.
    :param data_target_net: optional NetworkSpaceTimeData array.  If supplied, the linkage indices are between
    data_source and data_target, otherwise the two are set equal
    :param data_target_txy: as above but a EuclideanSpaceTimeData array
    :param chunksize: The size of an iteration chunk.
    :return: tuple (idx_array_source, idx_array_target),
    """
    ndata_source = data_source_net.ndata
    if data_source_txy:
        if data_source_txy.ndata != ndata_source:
            raise AttributeError("data_source_net and data_source_xy are different sizes.")
    else:
        # create Cartesian version from network version
        data_source_txy = data_source_net.time
        data_source_txy = data_source_txy.adddim(data_source_net.space.to_cartesian(), type=CartesianSpaceTimeData)

    if data_target_net is not None:
        ndata_target = data_target_net.ndata
        if data_target_txy:
            if data_target_txy.ndata != ndata_target:
                raise AttributeError("data_target_net and data_target_xy are different sizes.")
        else:
            # create Cartesian version from network version
            data_target_txy = data_target_net.time
            data_target_txy = data_target_txy.adddim(data_target_net.space.to_cartesian(), type=CartesianSpaceTimeData)
        n_pair = ndata_source * ndata_target

    else:
        # self-linkage case
        if data_target_txy is not None:
            raise AttributeError("Cannot supply data_target_txy without data_target_net")
        data_target_net = data_source_net
        data_target_txy = data_source_txy
        n_pair = ndata_source * (ndata_source - 1) / 2

    # quick Euclidean scan
    idx_i, idx_j = linkages(
        data_source_txy,
        linkage_fun,
        data_target=data_target_txy,
        chunksize=chunksize,
        remove_coincident_pairs=remove_coincident_pairs)

    print "Eliminated %d / %d links by quick Euclidean scan (%.1f %%)" % (
        n_pair - idx_i.size,
        n_pair,
        100. * (1 - idx_i.size / float(n_pair))
    )

    if not idx_i.size:
        return np.array([]), np.array([]), np.array([]), np.array([])

    link_i = []
    link_j = []
    dt = []
    dd = []

    chunksize = min(chunksize, idx_i.size)

    for k in range(0, idx_i.size, chunksize):
        # get chunk indices
        i = idx_i.flat[k:(k + chunksize)]
        j = idx_j.flat[k:(k + chunksize)]

        # recompute dt and dd, this time using NETWORK DISTANCE
        this_dt = (data_target_net.time.getrows(j) - data_source_net.time.getrows(i)).toarray(0)
        this_dd = (data_target_net.space.getrows(j).distance(data_source_net.space.getrows(i))).toarray(0)

        # reapply the linkage threshold function
        mask_net = linkage_fun(this_dt, this_dd) & (this_dt > 0)

        link_i.extend(i[mask_net])
        link_j.extend(j[mask_net])
        dt.extend(this_dt[mask_net])
        dd.extend(this_dd[mask_net])

    return np.array(link_i), np.array(link_j), np.array(dt), np.array(dd)

from data.models import NetworkData
from collections import defaultdict
from network.streetnet import Edge


def get_next_node(edge, node):
    """ Get the ID of the node that is NOT node """
    return edge.orientation_pos if edge.orientation_pos != node else edge.orientation_neg


def network_walker(net_obj,
                   source_node=None,
                   max_distance=None,
                   repeat_edges=False,
                   initial_exclusion=None,
                   verbose=False):
    """
    Generator, yielding (path, distance, edge) tuples giving the path taken, total distance travelled and
    edge of a network walker.
    :param net_obj:
    :param source_node: Optional. The node to start at. Otherwise the first listed node will be used.
    :param max_distance: Optional. The maximum distance to travel. Any edge that BEGINS within this distance of the
    start node will be returned.
    :param repeat_edges: If True then the walker will cover the same edges more than once, provided that doing so
    doesn't result in a loop.  Results in many more listed paths. Required for KDE normalisation, but should be set
    to False for search and sampling operations.
    :param initial_exclusion: Optionally provide the ID of a node to exclude when choosing the first 'step'. This is
    necessary when searching from a NetPoint.
    """
    logger = logging.getLogger("network_walker.logger")
    logger.handlers = []  # make sure logger has no handlers to begin with
    if verbose:
        logger.setLevel(logging.DEBUG)
        logger.addHandler(logging.StreamHandler())
    else:
        logger.addHandler(logging.NullHandler())

    if initial_exclusion is not None and source_node is None:
        # this doesn't make any sense
        raise AttributeError("If initial_exclusion node is supplied, must also supply the source_node")

    if source_node is None:
        source_node = net_obj.nodes()[0]

    edges_seen = {}  # only used if repeat_edges = False

    #A list which monitors the current state of the path
    current_path = [source_node]

    # A list that records the distance to each step on the current path. This is initially equal to zero
    dist = [0]

    # A stack that lists the next nodes to be searched. Each item in the stack
    # is a list of edges accessible from the previous node, excluding a reversal.

    stack = [net_obj.next_turn(source_node, exclude_nodes=initial_exclusion)]  # list of lists

    # keep a tally of generation number
    count = 0

    #The stack will empty when the source has been exhausted
    while stack:

        logger.info("Stack has length %d. Picking from top of the stack.", len(stack))

        if not len(stack[-1]):
            # no more nodes to search from previous edge
            # remove the now empty list from the stack
            stack.pop()
            #Backtrack to the previous position on the current path
            current_path.pop()
            #Adjust the distance list to represent the new state of current_path too
            dist.pop()

            logger.info("Options exhausted. Backtracking...")

            # skip to next iteration
            continue

        # otherwise, grab and remove the next edge to search
        this_edge = stack[-1].pop()

        if not repeat_edges:
            if this_edge in edges_seen:
                logger.info("Have already seen this edge on iteration %d, so won't repeat it.", edges_seen[this_edge])
                # skip to next iteration
                continue
            else:
                logger.info("This is the first time we've walked this edge")
                edges_seen[this_edge] = count

        logger.info("*** Generation %d ***", count)
        count += 1
        yield (list(current_path), dist[-1], this_edge)

        logger.info("Walking edge %s", this_edge)

        # check whether next node is within reach (if max_distance has been supplied)
        if max_distance is not None:
            dist_to_next_node = dist[-1] + this_edge.length
            if dist_to_next_node > max_distance:
                logger.info("Walking to the end of this edge is too far (%.2f), so won't explore further.",
                            dist_to_next_node)
                continue

        # Add the next node's edges to the stack if it hasn't already been visited
        # TODO: if we need to support loops, then skip this checking stage?
        previous_node = current_path[-1]
        node = get_next_node(this_edge, previous_node)
        # has this node been visited already?
        if node not in current_path:
            logger.info("Haven't visited this before, so adding to the stack.")
            stack.append(net_obj.next_turn(node, exclude_nodes=[previous_node]))
            current_path.append(node)
            dist.append(dist[-1] + this_edge.length)
            logger.info("We are now distance %.2f away from the source point", dist[-1])
        else:
            logger.info("We were already here on iteration %d so ignoring it", (current_path.index(node) + 1))


def network_walker_from_net_point(net_obj,
                                  net_point,
                                  max_distance=None,
                                  repeat_edges=False,
                                  verbose=False):
    """
    Very similar to network_walker, but support starting from a NetPoint rather than a node on the network itself.
    Essentially this involves walking both ways from the point (i.e. start at pos then neg node), avoiding doubling back
    in both cases. Also yield the initial edge for convenience.
    All inputs same as network_walker.
    :param net_obj:
    :param net_point:
    :param max_distance:
    :param repeat_edges:
    :param verbose:
    :return:
    """
    g_pos = network_walker(net_obj,
                           source_node=net_point.edge.orientation_pos,
                           max_distance=max_distance - net_point.distance_positive,
                           initial_exclusion=net_point.edge.orientation_neg,
                           repeat_edges=repeat_edges,
                           verbose=verbose)
    g_neg = network_walker(net_obj,
                           source_node=net_point.edge.orientation_neg,
                           max_distance=max_distance - net_point.distance_negative,
                           initial_exclusion=net_point.edge.orientation_pos,
                           repeat_edges=repeat_edges,
                           verbose=verbose)

    # first edge to generate is always the edge on which net_point is located
    yield ([], 0., net_point.edge)

    for g in [g_pos, g_neg]:
        for t in g:
            yield t


def network_walker_uniform_sample_points(net_obj, interval, source_node=None):
    """
    Generate NetPoints uniformly along the network with the supplied interval
    :param net_obj: StreetNet instance
    :param interval: Distance between points
    :param source_node: Optionally specify the node to start at. This will affect the outcome.
    :return:
    """
    g = network_walker(net_obj, source_node=source_node, repeat_edges=False)
    points = OrderedDict()
    n_per_edge = OrderedDict()
    for e in net_obj.edges():
        points[e] = None
        n_per_edge[e] = None

    # points = []
    # n_per_edge = []
    for path, dist, edge in g:
        el = edge.length

        # next point location
        npl = interval - dist % interval

        # distances along to deposit points
        point_dists = np.arange(npl, el, interval)

        if not point_dists.size:
            # this edge is too short - just place one point at the centroid
            # points.append(edge.centroid)
            # n_per_edge.append(1)
            points[edge] = [edge.centroid]
            n_per_edge[edge] = 1
            continue
        else:
            n_per_edge[edge] = point_dists.size
            # n_per_edge.append(point_dists.size)

        # create the points
        on = path[-1]
        op = get_next_node(edge, path[-1])

        points[edge] = []
        for pd in point_dists:
            node_dist = {
                on: pd,
                op: el - pd,
            }
            # points.append(NetPoint(net_obj, edge, node_dist))
            points[edge].append(NetPoint(net_obj, edge, node_dist))

    points = NetworkData(reduce(operator.add, points.values()))
    # res = NetworkData(points)
    # n_per_edge = np.array(n_per_edge)
    n_per_edge = np.array(n_per_edge.values())

    return points, n_per_edge


def network_paths_source_targets(net_obj, source, targets, max_search_distance, verbose=False):
    target_points = NetworkData(targets)
    paths = defaultdict(list)

    g = network_walker_from_net_point(net_obj,
                                      source,
                                      max_distance=max_search_distance,
                                      repeat_edges=True,
                                      verbose=verbose)

    target_distance = target_points.euclidean_distance(NetworkData([source] * target_points.ndata))
    reduced_target_idx = np.where(target_distance.toarray(0) <= max_search_distance)[0]
    reduced_targets = target_points.getrows(reduced_target_idx)
    # print "Reduced targets from %d to %d using Euclidean cutoff" % (target_points.ndata, reduced_targets.ndata)

    for path, dist, edge in g:
        # test whether any targets lie on the new edge
        for i, t in enumerate(reduced_targets.toarray(0)):
            if t.edge == edge:
                # get distance from current node to this point
                if not len(path):
                    # this only happens at the starting edge
                    dist_between = (t - source).length
                else:
                    # all other situations
                    dist_along = t.node_dist[path[-1]]
                    dist_between = dist + dist_along
                # print "Target %d is on this edge at a distance of %.2f" % (reduced_target_idx[i], dist_between)
                if dist_between <= max_search_distance:
                    # print "Adding target %d to paths" % reduced_target_idx[i]
                    paths[reduced_target_idx[i]].append((list(path), dist_between))

    return paths
