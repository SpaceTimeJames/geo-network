import logging
import operator
import numpy as np
from collections import OrderedDict, defaultdict
from point import NetPoint, NetPointArray
from path import NetPath


def get_next_node(edge, node):
    """ Get the ID of the node that is NOT node """
    return edge.orientation_pos if edge.orientation_pos != node else edge.orientation_neg


class NetworkWalker(object):
    """
    Walks the network from a starting edge or node, generating all possible paths, optionally subject to a maximum
    distance parameter.
    """
    def __init__(self, net_obj, targets,
                 max_distance=None,
                 max_split=None,
                 repeat_edges=True,
                 verbose=False,
                 logger=None):
        self.net_obj = net_obj
        self.targets = NetPointArray(targets)
        self.max_distance = max_distance
        self.max_split = max_split
        self.repeat_edges = repeat_edges

        # logging
        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger("NetworkWalker")
            self.logger.handlers = []  # make sure logger has no handlers to begin with
            if verbose:
                self.logger.addHandler(logging.StreamHandler())
            else:
                self.logger.addHandler(logging.NullHandler())

        if verbose:
            self.verbose = True
            self.logger.setLevel(logging.DEBUG)
        else:
            self.verbose = False
            self.logger.setLevel(logging.INFO)

        # this dictionary keeps track of the walks already carried out
        self.cached_walks = {}
        self.cached_source_target_paths = {}

    @property
    def n_targets(self):
        return len(self.targets)

    def caching_func(self, key, gen):
        """ caches then yields the results of the supplied generator, using supplied key """
        self.cached_walks[key] = []
        for t in gen:
            self.cached_walks[key].append(t)
            yield t

    def walk_from_node(self, start):
        g = network_walker(self.net_obj,
                           start,
                           max_distance=self.max_distance,
                           max_split=self.max_split,
                           repeat_edges=self.repeat_edges,
                           logger=self.logger)

        return self.caching_func(start, g)

    def walk_from_net_point(self, start):
        # try to retrieve from the cache
        try:
            return (t for t in self.cached_walks[start])
        except KeyError:
            g = network_walker_from_net_point(self.net_obj,
                                              start,
                                              max_distance=self.max_distance,
                                              max_split=self.max_split,
                                              repeat_edges=self.repeat_edges,
                                              logger=self.logger,
                                              verbose=self.verbose)

            return self.caching_func(start, g)

    def walker(self, start=None):
        if start is None:
            # default behaviour: start at the first node
            start = self.net_obj.nodes()[0]

        # try to retrieve from the cache
        try:
            return (t for t in self.cached_walks[start])
        except KeyError:
            # compute it from scratch
            if hasattr(start, 'edge'):
                return self.walk_from_net_point(start)
            else:
                return self.walk_from_node(start)

    def source_to_targets(self, start, max_distance=None):
        """
        Starting at the (node|NetPoint) start, find all paths to the targets, subject to the usual constraints
        :param start:
        :param max_distance: Optionally provide a new max_distance.
        :return:
        """
        max_distance = max_distance or self.max_distance
        try:
            # look for cached result
            md, cached = self.cached_source_target_paths[start]
            # check the max_distance associated with the cached value
            # equal: simply return the cache
            # greater: filter the cache for results that meet the current max_distance
            # less: recompute the result and cache
            if md == max_distance:
                self.logger.info("source_to_targets called with the SAME max_distance; returning cached result.")
                return cached
            elif md > max_distance:
                self.logger.info("source_to_targets called with a LOWER max_distance; filtering cached result.")
                new_paths = {}
                for k, x in cached.iteritems():
                    filtered = [t for t in x if t.distance_total <= max_distance]
                    if len(filtered):
                        new_paths[k] = filtered
                return new_paths
            else:
                self.logger.info("source_to_targets called with a HIGHER max_distance; dropping cached result and recomputing.")
                self.cached_source_target_paths.pop(start)
        except KeyError:
            # compute it from scratch
            self.logger.info("No cached result retrieved; computing result from scratch.")

        paths = network_paths_source_targets(self.net_obj,
                                             start,
                                             self.targets,
                                             max_distance,
                                             max_split=self.max_split,
                                             verbose=self.verbose,
                                             logger=self.logger)
        self.cached_source_target_paths[start] = (max_distance, paths)
        return paths


def network_walker(net_obj,
                   source_node=None,
                   max_distance=None,
                   max_split=None,
                   repeat_edges=False,
                   initial_exclusion=None,
                   verbose=False,
                   logger=None):
    """
    Generator, yielding (path, distance, edge) tuples giving the path taken, total distance travelled and
    edge of a network walker.
    :param net_obj:
    :param source_node: Optional. The node to start at. Otherwise the first listed node will be used.
    :param max_distance: Optional. The maximum distance to travel. Any edge that BEGINS within this distance of the
    start node will be returned.
    :param max_split: Optional. The maximum number of times the path may branch. Useful because every branch point
    reduces the density of a kernel by a factor of (degree - 1), so at some point we can discount paths.
    :param repeat_edges: If True then the walker will cover the same edges more than once, provided that doing so
    doesn't result in a loop.  Results in many more listed paths. Required for KDE normalisation, but should be set
    to False for search and sampling operations.
    :param initial_exclusion: Optionally provide the ID of an edge to exclude when choosing the first 'step'. This is
    necessary when searching from a NetPoint.
    :param verbose: If True, log debug info relating to the algorithm
    :param logger: If supplied, use this logger and ignore the value of verbose.
    """
    if logger is None:
        logger = logging.getLogger("network_walker.logger")
        logger.handlers = []  # make sure logger has no handlers to begin with
        if verbose:
            logger.setLevel(logging.DEBUG)
            logger.addHandler(logging.StreamHandler())
        else:
            logger.addHandler(logging.NullHandler())
    else:
        if verbose:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)

    if initial_exclusion is not None and source_node is None:
        # this doesn't make any sense
        raise AttributeError("If initial_exclusion node is supplied, must also supply the source_node")

    if source_node is None:
        source_node = net_obj.nodes()[0]

    edges_seen = {}  # only used if repeat_edges = False

    # A list which monitors the current state of the path
    current_path = [source_node]
    current_splits = [max(net_obj.degree(source_node) - 1., 1.)]

    # A list that records the distance to each step on the current path. This is initially equal to zero
    dist = [0]

    # A stack that lists the next nodes to be searched. Each item in the stack
    # is a list of edges accessible from the previous node, excluding a reversal.

    stack = [net_obj.next_turn(source_node, exclude_edges=initial_exclusion)]  # list of lists

    # keep a tally of generation number
    count = 0

    #The stack will empty when the source has been exhausted
    while stack:

        logger.debug("Stack has length %d. Picking from top of the stack.", len(stack))

        if not len(stack[-1]):
            # no more nodes to search from previous edge
            # remove the now empty list from the stack
            stack.pop()
            #Backtrack to the previous position on the current path
            current_path.pop()
            current_splits.pop()
            #Adjust the distance list to represent the new state of current_path too
            dist.pop()

            logger.debug("Options exhausted. Backtracking...")

            # skip to next iteration
            continue

        # otherwise, grab and remove the next edge to search
        this_edge = stack[-1].pop()

        if not repeat_edges:
            if this_edge in edges_seen:
                logger.debug("Have already seen this edge on iteration %d, so won't repeat it.", edges_seen[this_edge])
                # skip to next iteration
                continue
            else:
                logger.debug("This is the first time we've walked this edge")
                edges_seen[this_edge] = count

        logger.debug("*** Generation %d ***", count)
        count += 1
        this_path = NetPath(
            net_obj,
            start=source_node,
            end=current_path[-1],
            nodes=list(current_path),
            distance=dist[-1],
            split=current_splits[-1])
        yield this_path, this_edge

        logger.debug("Walking edge %s", this_edge)

        # check whether next node is within reach (if max_distance has been supplied)
        if max_distance is not None:
            dist_to_next_node = dist[-1] + this_edge.length
            if dist_to_next_node > max_distance:
                logger.debug("Walking to the end of this edge is too far (%.2f), so won't explore further.",
                            dist_to_next_node)
                continue

        # Add the next node's edges to the stack if it hasn't already been visited
        # TODO: if we need to support loops, then skip this checking stage?
        previous_node = current_path[-1]
        node = get_next_node(this_edge, previous_node)

        # check whether the number of branches is below tolerance (if max_splits has been supplied)
        next_splits = current_splits[-1] * max(net_obj.degree(node) - 1., 1.)
        if max_split is not None and next_splits > max_split:
            logger.debug("The next node has too many branches, so won't explore further.")
            continue

        # has this node been visited already?
        if node not in current_path:
            logger.debug("Haven't visited this before, so adding to the stack.")
            stack.append(net_obj.next_turn(node, exclude_edges=[this_edge.fid]))
            current_path.append(node)
            current_splits.append(next_splits)
            dist.append(dist[-1] + this_edge.length)
            logger.debug("We are now distance %.2f away from the source point", dist[-1])
        else:
            logger.debug("We were already here on iteration %d so ignoring it", (current_path.index(node) + 1))


def network_walker_from_net_point(net_obj,
                                  net_point,
                                  max_distance=None,
                                  max_split=None,
                                  repeat_edges=False,
                                  verbose=False,
                                  logger=None):
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
    d_pos = net_point.distance_positive
    d_neg = net_point.distance_negative

    # first edge to generate is always the edge on which net_point is located
    # the path in this case is null, and shouldn't be used.
    this_path = NetPath(
        net_obj,
        start=net_point,
        end=net_point,
        nodes=[],
        distance=0.,
        split=1.)
    yield this_path, net_point.edge

    ## TODO: add support for max_distance=None

    if max_distance is not None and max_distance - d_pos > 0:
        g_pos = network_walker(net_obj,
                               source_node=net_point.edge.orientation_pos,
                               max_distance=max_distance - d_pos,
                               max_split=max_split,
                               initial_exclusion=net_point.edge.fid,
                               repeat_edges=repeat_edges,
                               verbose=verbose,
                               logger=logger)
        for (path, edge) in g_pos:
            # replace start node with true start
            path.start = net_point
            # add distance already walked
            path.distance_total += d_pos
            yield path, edge

    if max_distance is not None and max_distance - d_neg > 0:
        g_neg = network_walker(net_obj,
                               source_node=net_point.edge.orientation_neg,
                               max_distance=max_distance - d_neg,
                               max_split=max_split,
                               initial_exclusion=net_point.edge.fid,
                               repeat_edges=repeat_edges,
                               verbose=verbose,
                               logger=logger)
        for (path, edge) in g_neg:
            # replace start node with true start
            path.start = net_point
            # add distance already walked
            path.distance_total += d_neg
            yield path, edge


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

    for path, edge in g:
        el = edge.length

        # next point location
        # npl = interval - dist % interval
        npl = interval - path.distance_total % interval

        # distances along to deposit points
        point_dists = np.arange(npl, el, interval)

        if not point_dists.size:
            # this edge is too short - just place one point at the centroid
            points[edge] = [edge.centroid]
            n_per_edge[edge] = 1
            continue
        else:
            n_per_edge[edge] = point_dists.size

        # create the points
        on = path.nodes[-1]
        op = get_next_node(edge, path.nodes[-1])

        points[edge] = []
        for pd in point_dists:
            node_dist = {
                on: pd,
                op: el - pd,
            }
            points[edge].append(NetPoint(net_obj, edge, node_dist))

    points = NetPointArray(reduce(operator.add, points.values()))
    n_per_edge = np.array(n_per_edge.values())

    return points, n_per_edge


def network_walker_fixed_distance(net_obj,
                                  starting_net_point,
                                  distance,
                                  verbose=False):
    """
    Generate NetPoints at fixed distance from starting point
    :param net_obj: StreetNet instance
    :param starting_net_point: Starting point
    :param distance: Distance to walk.
    :return:
    """
    g = network_walker_from_net_point(net_obj,
                                      starting_net_point,
                                      repeat_edges=True,
                                      max_distance=distance,
                                      verbose=verbose)
    end_points = []
    paths = []

    for path, edge in g:
        el = edge.length
        if not len(path.nodes):
            # only true for the starting edge
            # if either node is greater than max_distance away, add a net point
            if starting_net_point.distance_negative > distance:
                node_dist = {
                    edge.orientation_neg: starting_net_point.distance_negative - distance,
                }
                end_points.append(NetPoint(net_obj, edge, node_dist))
                paths.append(path)
            if starting_net_point.distance_positive > distance:
                node_dist = {
                    edge.orientation_pos: starting_net_point.distance_positive - distance,
                }
                end_points.append(NetPoint(net_obj, edge, node_dist))
                paths.append(path)
            continue

        next_node = get_next_node(edge, path.nodes[-1])

        if path.distance_total + el <= distance:
            if net_obj.g.degree(next_node) == 1:
                # terminal node: stop here
                node_dist = {
                    path.nodes[-1]: el,
                    next_node: 0.,
                }
            else:
                continue
        else:
            # how far along this edge can we walk?
            da = distance - path.distance_total
            # construct point
            node_dist = {
                path.nodes[-1]: da,
                next_node: el - da
            }
        netp = NetPoint(net_obj, edge, node_dist)
        end_points.append(netp)
        paths.append(path)

    return end_points, paths


def network_paths_source_targets(net_obj,
                                 source,
                                 targets,
                                 max_search_distance,
                                 max_split=None,
                                 verbose=False,
                                 logger=None):
    if logger is None:
        logger = logging.getLogger('null')
        logger.handlers = []
        logger.addHandler(logging.NullHandler())
    target_points = NetPointArray(targets)
    paths = defaultdict(list)

    g = network_walker_from_net_point(net_obj,
                                      source,
                                      max_distance=max_search_distance,
                                      max_split=max_split,
                                      repeat_edges=True,
                                      verbose=verbose,
                                      logger=logger)

    # Cartesian filtering by nodes
    node_coords_pos = np.array([t.edge.node_pos_coords for t in target_points])
    node_coords_neg = np.array([t.edge.node_neg_coords for t in target_points])

    # target_nodes_pos = CartesianData([t.edge.node_pos_coords for t in target_points.toarray(0)])
    # target_nodes_neg = CartesianData([t.edge.node_neg_coords for t in target_points.toarray(0)])

    # Find the targets on the source edge and include these explicitly.
    # This is required for longer edges, where neither of the edge nodes are within max_search distance
    on_this_edge = np.array([t.edge == source.edge for t in target_points])
    logger.debug("Found %d points on the starting edge" % on_this_edge.sum())

    target_distance_pos = np.sum(
        (node_coords_pos - source.cartesian_coords) ** 2,
        axis=1
    ) ** 0.5

    target_distance_neg = np.sum(
        (node_coords_neg - source.cartesian_coords) ** 2,
        axis=1
    ) ** 0.5

    reduced_target_idx = np.where(
        (target_distance_pos <= max_search_distance) |
        (target_distance_neg <= max_search_distance) |
        on_this_edge
    )[0]
    reduced_targets = target_points[reduced_target_idx]
    logger.debug("Initial filtering reduces number of targets from {0} to {1}".format(
        len(target_points),
        len(reduced_targets)))


    # cartesian filtering by NetPoint
    # source_xy_tiled = CartesianData([source.cartesian_coords] * target_points.ndata)
    # target_distance = target_points.to_cartesian().distance(source_xy_tiled)
    # reduced_target_idx = np.where(target_distance.toarray(0) <= max_search_distance)[0]
    # reduced_targets = target_points.getrows(reduced_target_idx)

    # ALL targets kept
    # reduced_targets = target_points
    # reduced_target_idx = range(target_points.ndata)

    for path, edge in g:
        # test whether any targets lie on the new edge
        for i, t in enumerate(reduced_targets):
            if t.edge == edge:
                # get distance from current node to this point
                if not len(path.nodes):
                    # this only happens at the starting edge
                    dist_between = (t - source).length
                else:
                    # all other situations
                    dist_along = t.node_dist[path.nodes[-1]]
                    dist_between = path.distance_total + dist_along
                logger.debug("Target %d is on this edge at a distance of %.2f" % (reduced_target_idx[i], dist_between))
                if dist_between <= max_search_distance:
                    logger.debug("Adding target %d to paths" % reduced_target_idx[i])
                    this_path = NetPath(
                        net_obj,
                        start=path.start,
                        end=t,
                        nodes=list(path.nodes),
                        distance=dist_between,
                        split=path.splits_total)
                    paths[reduced_target_idx[i]].append(this_path)

    return dict(paths)
