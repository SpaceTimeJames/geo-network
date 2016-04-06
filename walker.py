import logging

def get_next_node(edge, node):
    """ Get the ID of the node that is NOT node """
    return edge.orientation_pos if edge.orientation_pos != node else edge.orientation_neg


class NetworkWalkerBase(object):
    """
    Walks the network from a starting edge or node, generating all possible paths, optionally subject to a maximum
    distance parameter.
    """
    def __init__(self, net,
                 max_split=1024,
                 repeat_edges=True,
                 reflect=True,
                 allow_cycles=True,
                 verbose=False,
                 logger=None):

        self.net = net
        self.max_split = max_split
        self.repeat_edges = repeat_edges
        self.reflect = reflect
        self.allow_cycles = allow_cycles

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

    def caching_func(self, key, gen, **attrs):
        """ caches then yields the results of the supplied generator, using supplied key """
        self.cached_walks[key] = attrs
        self.cached_walks[key]['paths'] = []

        for t in gen:
            self.cached_walks[key]['paths'].append(t)
            yield t

    def _walk_from_node(self,
                       start_node,
                       initial_exclude_edge=None,
                       max_distance=None):
        g = network_walker(self.net,
                           start_node,
                           max_distance=max_distance,
                           max_split=self.max_split,
                           repeat_edges=self.repeat_edges,
                           reflect=self.reflect,
                           allow_cycles=self.allow_cycles,
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

    # The stack will empty when the source has been exhausted
    while stack:

        logger.debug("Stack has length %d. Picking from top of the stack.", len(stack))

        if not len(stack[-1]):
            # no more nodes to search from previous edge
            # remove the now empty list from the stack
            stack.pop()
            # Backtrack to the previous position on the current path
            current_path.pop()
            current_splits.pop()
            # Adjust the distance list to represent the new state of current_path too
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