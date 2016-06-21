__author__ = 'gabriel'
import numpy as np
from data.models import CartesianSpaceTimeData, NetworkData, CartesianData
import logging
from streetnet import Edge, NetPath
from point import NetPoint
from collections import OrderedDict, defaultdict
import operator


uint_dtypes = [(t, np.iinfo(t)) for t in (
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
)]

int_dtypes = [(t, np.iinfo(t)) for t in (
    np.int8,
    np.int16,
    np.int32,
    np.int64,
)]


def numpy_most_compact_int_dtype(arr):
    """
    Compress supplied array of integers as much as possible by changing the dtype
    :param arr:
    :return:
    """
    if np.any(arr < 0):
        dtypes = int_dtypes
    else:
        dtypes = uint_dtypes

    arr_max = arr.max()  ## FIXME: max ABS value
    for t, ii in dtypes:
        if arr_max <= ii.max:
            return arr.astype(t)

    raise ValueError("Unable to find a suitable datatype")



def linkage_func_separable(max_t, max_d):
    def func(dt, dd):
        return (dt <= max_t) & (dd <= max_d)
    return func



def pairwise_differences_indices(n):

    dtypes = [
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
    ]

    dtype = None
    # find appropriate datatype
    for d in dtypes:
        if np.iinfo(d).max >= (n - 1):
            dtype = d
            break

    if not dtype:
        raise MemoryError("Unable to index an array this large.")

    idx_i = np.zeros(n* (n - 1) / 2, dtype=dtype)
    idx_j = np.zeros_like(idx_i)

    tally = 0
    for i in range(n):
        idx_i[tally:(tally + n - i - 1)] = np.ones(n - i - 1, dtype=dtype) * i
        idx_j[tally:(tally + n - i - 1)] = np.arange(i + 1, n, dtype=dtype)
        tally += n - i - 1

    return idx_i, idx_j


def linkages(data_source,
             threshold_fun,
             data_target=None,
             chunksize=2**18,
             remove_coincident_pairs=False,
             time_gte_zero=True):
    """
    Compute the indices of datapoints that cause threshold_fun to return True
    The sign convention is (target - source).  Distances are euclidean.
    :param data_source: EuclideanSpaceTimeData array of source data.  Must be sorted by time ascending.
    :param threshold_fun: function that operates on the DataArray (delta_t, delta_d) returning True for linked points
    NB: also require that delta_t > 0.
    :param data_target: optional EuclideanSpaceTimeData array.  If supplied, the linkage indices are between
    data_source and data_target, otherwise the two are set equal
    :param chunksize: The size of an iteration chunk.
    :param remove_coincident_pairs: If True, remove links between pairs of crimes with Delta d == 0
    :param time_gte_zero: If True, enforce dt > 0 in addition to threshold function. This is the usual desired behaviour
    for space-time point processes, but is not required for some spatial only analyses.
    :return: tuple (idx_array_source, idx_array_target),
    """
    ndata_source = data_source.ndata
    if data_target is not None:
        ndata_target = data_target.ndata
        chunksize = min(chunksize, ndata_source * ndata_target)
        idx_i, idx_j = np.meshgrid(range(ndata_source), range(ndata_target), copy=False)
    else:
        # self-linkage
        data_target = data_source
        chunksize = min(chunksize, ndata_source * (ndata_source - 1) / 2)
        idx_i, idx_j = pairwise_differences_indices(ndata_source)

    link_i = []
    link_j = []

    for k in range(0, idx_i.size, chunksize):
        i = idx_i.flat[k:(k + chunksize)]
        j = idx_j.flat[k:(k + chunksize)]
        dt = (data_target.time.getrows(j) - data_source.time.getrows(i))
        dd = (data_target.space.getrows(j).distance(data_source.space.getrows(i)))
        if time_gte_zero:
            mask = threshold_fun(dt, dd) & (dt > 0)
        else:
            mask = threshold_fun(dt, dd)
        if remove_coincident_pairs:
            mask[dd.toarray() == 0] = False
        mask = mask.toarray(0)
        link_i.extend(i[mask])
        link_j.extend(j[mask])

    return np.array(link_i), np.array(link_j)


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
        this_dd = (data_target_net.space.getrows(j).distance(data_source_net.space.getrows(i))).toarray(0)  # RATE-LIMIT

        # reapply the linkage threshold function
        mask_net = linkage_fun(this_dt, this_dd) & (this_dt > 0)

        link_i.extend(i[mask_net])
        link_j.extend(j[mask_net])
        dt.extend(this_dt[mask_net])
        dd.extend(this_dd[mask_net])

    return np.array(link_i), np.array(link_j), np.array(dt), np.array(dd)


def network_point_coverage(net, dx=None, include_nodes=True):
    '''
    Produce a series of semi-regularly-spaced points on the supplied network.
    :param net: Network
    :param dx: Optional spacing between points, otherwise this is automatically selected
    :param include_nodes: If True, points are added at node locations too
    :return: - NetworkData array of NetPoints, ordered by edge
             - length E array of indices. Each gives the number of points in that edge
    '''

    # small delta to avoid errors
    eps = 1e-6

    ## temp set dx with a constant
    xy = []
    cd = []
    edge_count = []
    dx = dx or 1.

    for edge in net.edges():
        this_xy = []
        this_cd = []
        n_pt = int(np.math.ceil(edge['length'] / float(dx)))
        interp_lengths = np.linspace(eps, edge['length'] - eps, n_pt)
        # interpolate along linestring
        ls = edge['linestring']
        interp_pts = [ls.interpolate(t) for t in interp_lengths]

        for i in range(interp_lengths.size):
            this_xy.append((interp_pts[i].x, interp_pts[i].y))
            node_dist = {
                edge['orientation_neg']: interp_lengths[i],
                edge['orientation_pos']: edge['length'] - interp_lengths[i],
            }
            this_cd.append(NetPoint(net, edge, node_dist))
        xy.extend(this_xy)
        cd.extend(this_cd)
        edge_count.append(interp_lengths.size)

    return NetworkData(cd), edge_count