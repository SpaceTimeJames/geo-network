import logging
from streetnet import NetPath

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
                           initial_exclusion=initial_exclude_edge,
                           reflect=self.reflect,
                           allow_cycles=self.allow_cycles,
                           logger=self.logger)

        return self.caching_func(start_node, g, max_distance=max_distance, initial_exclusion=initial_exclude_edge)

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
                   repeat_edges=True,
                   initial_exclusion=None,
                   reflect=True,
                   allow_cycles=True,
                   verbose=False,
                   logger=None):
    """
    Generator, yielding (path, distance, edge) tuples giving the path taken, total distance travelled and
    edge of a network walker.
    NB: certain combinations of parameters will result in an infinite generator, e.g. enabling cycles and providing no
    max distance or split count
    :param net_obj:
    :param source_node: Optional. The node to start at. Otherwise the first listed node will be used.
    :param max_distance: Optional. The maximum distance to travel. Any edge that BEGINS within this distance of the
    start node will be returned.
    :param max_split: Optional. The maximum number of times the path may branch. Useful because every branch point
    reduces the density of a kernel by a factor of (degree - 1), so at some point we can discount paths.
    :param repeat_edges: If True then the walker will cover the same edges more than once, provided that doing so
    doesn't result in a loop.  Results in many more listed paths. Required for KDE normalisation, but should be set
    to False for search and sampling operations.
    :param reflect: If True (default), turn around at single degree nodes and continue walking
    :param allow_cycles: If True (default), permit walking around cycles.
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

    stack = [net_obj.next_turn(source_node, previous_edge_id=initial_exclusion)]  # list of lists

    # keep a tally of generation number
    count = 0
    count_edge = 0

    # The stack will empty when the source has been exhausted
    while stack:

        logger.debug("Stack has length %d. Picking from top of the stack.", len(stack))
        this_edge_list = stack.pop()
        this_path = current_path.pop()
        this_split = current_splits.pop()
        this_distance = dist.pop()
        logger.debug("We have travelled %.2f length units and split %d times to this point.",
                     (this_distance, this_split))

        if not len(this_edge_list):
            logger.debug("The edge list was empty, so there's nothing left to do from here.")
            # skip to next iteration
            continue

        for count_node, this_edge in enumerate(this_edge_list):

            if not repeat_edges:
                if this_edge in edges_seen:
                    logger.debug("We saw this edge on iteration %s and repeat_edges=False so won't repeat it.",
                                 edges_seen[this_edge])
                    # skip to next iteration
                    continue
                else:
                    logger.debug("This is the first time we've walked this edge")
                    edges_seen[this_edge] = '%d.%d' % (count_edge, count_node)

            logger.debug("*** Edge %d ***", count_edge)
            count_edge += 1
            out_path = NetPath(
                net_obj,
                start=source_node,
                end=this_path[-1],
                nodes=this_path,
                distance=this_distance,
                split=current_splits[-1])
            yield out_path, this_edge

            # Add the next edges to be explored from the end of this edge
            previous_node = this_path[-1]
            node = get_next_node(this_edge, previous_node)

            logger.debug("On edge %s. The next node is %s.", this_edge, node)

            # check whether next node is within reach (if max_distance has been supplied)
            dist_to_next_node = this_distance + this_edge.length
            if max_distance is not None and dist_to_next_node > max_distance:
                logger.debug("This node is beyond max_distance (%.2f > %.2f), won't explore further.",
                             dist_to_next_node,
                             max_distance)
                continue

            # check whether the number of branches is below tolerance (if max_splits has been supplied)
            next_splits = this_split * max(net_obj.degree(node) - 1., 1.)
            if max_split is not None and next_splits > max_split:
                logger.debug("Path beyond this node is more branched than max_split (%d > %d), won't explore further.",
                             next_splits,
                             max_split)
                continue

            # check for cycles (if they are forbidden)
            # it's a REFLECTION if the previous edge is the same as the next
            if not allow_cycles and node in this_path:
                logger.debug("We have already passed through this node and cycles are not allowed.")
                continue

            next_edges = net_obj.next_turn(node, previous_edge_id=this_edge.fid, reflection=reflect)
            logger.debug("Appending %d new edges to the stack.", len(next_edges))
            stack.append(next_edges)
            current_path.append(this_path + [node])
            current_splits.append(next_splits)
            dist.append(dist_to_next_node)