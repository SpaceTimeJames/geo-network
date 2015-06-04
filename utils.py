__author__ = 'gabriel'
import numpy as np
from point_process.utils import linkages, linkage_func_separable
from data.models import CartesianSpaceTimeData
import logging


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


def network_walker(net_obj, source_node=None, max_distance=None, verbose=False):
    """
    Generator, yielding (path, distance, edge) tuples giving the path taken, total distance travelled and
    edge of a network walker.
    :param net_obj:
    :param source_node: Optional. The node to start at. Otherwise the first listed node will be used.
    :param max_distance: Optional. The maximum distance to travel. Any edge that BEGINS within this distance of the
    start node will be returned.
    """
    logger = logging.getLogger("network_walker.logger")
    logger.handlers = []  # make sure logger has no handlers to begin with
    if verbose:
        logger.setLevel(logging.DEBUG)
        logger.addHandler(logging.StreamHandler())
    else:
        logger.addHandler(logging.NullHandler())

    if source_node is None:
        source_node = net_obj.nodes()[0]

    #A list which monitors the current state of the path
    current_path = [source_node]

    # A list that records the distance to each step on the current path. This is initially equal to zero
    dist = [0]

    # A stack that lists the next nodes to be searched. Each item in the stack
    # is a list of edges accessible from the previous node, excluding a reversal.

    stack = [net_obj.next_turn(source_node)]  # list of lists

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


def network_walker_uniform_sample_points(net_obj, interval, source_node=None):
    """
    Generate NetPoints uniformly along the network with the supplied interval
    :param net_obj: StreetNet instance
    :param interval: Distance between points
    :param source_node: Optionally specify the node to start at. This will affect the outcome.
    :return:
    """
    if source_node is None:
        source_node = net_obj.nodes()[0]

    sample_points = []
    edges_visited = []

    #A list which monitors the current state of the path
    current_path = [source_node]

    # A list that records the distance to each step on the current path. This is initially equal to the distance from
    # the point to the node.
    dist = []

    # A stack that lists the next nodes to be searched. Each item in the stack
    # is a list of edges accessible from the previous node, excluding a reversal.

    stack = [net_obj.next_turn(source_node)]  # list of lists

    #The stack will empty when the source has been exhausted
    while stack:

        print "Stack has length %d. Picking from top of the stack." % len(stack)

        if not len(stack[-1]):
            # no more nodes to search from previous edge
            # remove the now empty list from the stack
            stack.pop()
            #Backtrack to the previous position on the current path
            current_path.pop()
            #Adjust the distance list to represent the new state of current_path too
            dist.pop()

            print "Options exhausted. Backtracking..."

            # skip to next iteration
            continue

        # otherwise, grab and remove the next edge to search
        this_edge = stack[-1].pop()

        print "Walking edge %s" % this_edge

        # Now test whether any targets lie on the new edge
        for i, t in enumerate(reduced_targets.toarray(0)):
            if t.edge == this_edge:
                # get distance from current node to this point
                dist_along = t.node_dist[current_path[-1]]
                dist_between = dist[-1] + dist_along
                print "Target %d is on this edge at a distance of %.2f" % (reduced_target_idx[i], dist_between)
                if dist_between <= cutoff:
                    print "Adding target %d to paths" % reduced_target_idx[i]
                    paths[reduced_target_idx[i]].append((list(current_path), dist_between))

        # Add the next node's edges to the stack if it hasn't already been visited and it's within reach
        # TODO: if we need to support loops, then skip this checking stage?
        if dist[-1] + this_edge.length <= cutoff:

            # find the ID of the next node - whichever node we are not at right now
            previous_node = node
            node = get_next_node(this_edge, previous_node)
            # has this node been visited already?
            if node not in current_path:
                print "Haven't visited this before, so adding to the stack."
                stack.append(net_obj.next_turn(node, exclude_nodes=[previous_node]))
                current_path.append(node)
                dist.append(dist[-1] + this_edge.length)
                print "We are now distance %.2f away from the source point" % dist[-1]
            else:
                print "We were already here on iteration %d so ignoring it" % (current_path.index(node) + 1)


def network_linkage_walker(net_obj, source_point, target_points, cutoff):
    """
    Find all paths within the cutoff distance between the single source point and any node within the target points.
    Strategy: the source point is linked to two nodes, + and -. Starting at each in turn, carry out an exploration,
    noting valid paths as we go. Similar argument for reaching the destination points - find the distance to the two
    connected nodes then test whether we can make it to the point itself.
    We can EITHER search for ONLY the shortest path, OR all paths within the cutoff, excluding loops.
    TODO: add include_loops parameter and support these too
    :param net_obj: StreetNet instance
    :param source_point:
    :param target_points:
    :param cutoff:
    :return:
    """
    target_points = NetworkData(target_points)
    paths = defaultdict(list)

    source_neg = source_point.edge.orientation_neg
    source_pos = source_point.edge.orientation_pos
    source_fid = source_point.edge.fid

    # build array of the possible starting and exclusion for the next_turn method
    # the exclusion node is always the other end of the edge, to avoid walking along it twice

    source_nodes = [
        source_point.edge.orientation_pos,
        source_point.edge.orientation_neg,
    ]

    ## TODO: incorporate directed routing option - means that starting at both source nodes may not be an option
    # if net_obj.directed: ...

    # source_edges = [source_point.edge]
    # add_rev = True
    # if net_obj.directed:
    #     # Try to get the reverse edge. If it doesn't exist, this will throw an IndexError
    #     ## TODO: haven't actually tested this yet
    #     try:
    #         edge_rev = net_obj.g_routing[source_neg][source_pos][source_fid]
    #     except IndexError:
    #         add_rev = False
    #
    # if add_rev:
    #     print "Adding reverse edge"
    #     source_edges.append(
    #         Edge(
    #             net_obj,
    #             orientation_neg=source_pos,
    #             orientation_pos=source_neg,
    #             fid=source_fid
    #         )
    #     )

    # quick filter of the targets to exclude any that are too far away based on Euclidean distance
    target_distance = target_points.euclidean_distance(NetworkData([source_point] * target_points.ndata))
    reduced_target_idx = np.where(target_distance.toarray(0) <= cutoff)[0]
    reduced_targets = target_points.getrows(reduced_target_idx)
    print "Reduced targets from %d to %d using Euclidean cutoff" % (target_points.ndata, reduced_targets.ndata)

    # before we do anything else, check whether there are any targets on the starting edge
    # If this is the reverse edge (root_n==1), we need to skip the
    # first search for targets - they have already been found
    for i, t in enumerate(reduced_targets.toarray(0)):
        if t.edge == source_point.edge:
            # test whether matching targets are sufficiently close and add them if they are
            dist_between = (t - source_point).length
            if dist_between <= cutoff:
                print "Adding target %d as it is on the starting edge." % reduced_target_idx[i]
                paths[reduced_target_idx[i]].append(([], dist_between))  # no path to record, hence empty array

    for node in source_nodes:
        print "***"
        print "Starting at node %s" % node
        print "***"

        #A list which monitors the current state of the path
        current_path = [node]

        # A list that records the distance to each step on the current path. This is initially equal to the distance from
        # the point to the node.
        dist = [source_point.node_dist[node]]

        if dist[0] > cutoff:
            # nothing to do - this node is too far away
            continue

        # A stack that lists the next nodes to be searched. Each item in the stack
        # is a list of edges accessible from the previous node, excluding a reversal.

        stack = [net_obj.next_turn(node, exclude_nodes=[get_next_node(source_point.edge, node)])]  # list of lists

        #The stack will empty when the source has been exhausted
        while stack:

            print "Stack has length %d. Picking from top of the stack." % len(stack)

            if not len(stack[-1]):
                # no more nodes to search from previous edge
                # remove the now empty list from the stack
                stack.pop()
                #Backtrack to the previous position on the current path
                current_path.pop()
                #Adjust the distance list to represent the new state of current_path too
                dist.pop()

                print "Options exhausted. Backtracking..."

                # skip to next iteration
                continue

            # otherwise, grab and remove the next edge to search
            this_edge = stack[-1].pop()

            print "Exploring edge %s" % this_edge

            # Now test whether any targets lie on the new edge
            for i, t in enumerate(reduced_targets.toarray(0)):
                if t.edge == this_edge:
                    # get distance from current node to this point
                    dist_along = t.node_dist[current_path[-1]]
                    dist_between = dist[-1] + dist_along
                    print "Target %d is on this edge at a distance of %.2f" % (reduced_target_idx[i], dist_between)
                    if dist_between <= cutoff:
                        print "Adding target %d to paths" % reduced_target_idx[i]
                        paths[reduced_target_idx[i]].append((list(current_path), dist_between))

            # Add the next node's edges to the stack if it hasn't already been visited and it's within reach
            # TODO: if we need to support loops, then skip this checking stage?
            if dist[-1] + this_edge.length <= cutoff:

                # find the ID of the next node - whichever node we are not at right now
                previous_node = node
                node = get_next_node(this_edge, previous_node)
                # has this node been visited already?
                if node not in current_path:
                    print "Haven't visited this before, so adding to the stack."
                    stack.append(net_obj.next_turn(node, exclude_nodes=[previous_node]))
                    current_path.append(node)
                    dist.append(dist[-1] + this_edge.length)
                    print "We are now distance %.2f away from the source point" % dist[-1]
                else:
                    print "We were already here on iteration %d so ignoring it" % (current_path.index(node) + 1)

    return paths
