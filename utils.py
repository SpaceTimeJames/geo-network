__author__ = 'gabriel'
import numpy as np
from point_process.utils import linkages, linkage_func_separable
from data.models import CartesianSpaceTimeData


#TODO: update this to use a function too, then implement that in hotspot.STNetworkBowers
def network_linkages(data_source_net,
                     max_t,
                     max_d,
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
    :param max_t: maximum time difference (minimum is always zero)
    :param max_d: maximum spatial distance
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
    link_fun = linkage_func_separable(max_t, max_d)
    idx_i, idx_j = linkages(
        data_source_txy,
        link_fun,
        data_target=data_target_txy,
        chunksize=chunksize,
        remove_coincident_pairs=remove_coincident_pairs)

    print "Eliminated %d / %d links by quick Euclidean scan (%.1f %%)" % (
        n_pair - idx_i.size,
        n_pair,
        100. * (1 - idx_i.size / float(n_pair))
    )

    if not idx_i.size:
        return np.array([]), np.array([])

    link_i = []
    link_j = []

    chunksize = min(chunksize, idx_i.size)

    for k in range(0, idx_i.size, chunksize):
        # get chunk indices
        i = idx_i.flat[k:(k + chunksize)]
        j = idx_j.flat[k:(k + chunksize)]

        # time difference is independent of spatial representation
        # dt = (data_target_net.time.getrows(j) - data_source_net.time.getrows(i)).toarray(0)

        # get Euclidean distance as a min bound
        # dd_euc = (data_target_txy.space.getrows(j).distance(data_source_txy.space.getrows(i))).toarray(0)
        # mask = (dt <= max_t) & (dt > 0.) & (dd_euc <= max_d)
        # if remove_coincident_pairs:
        #     mask &= (dd_euc != 0)

        # eliminate chunk indices
        # repeat for remaining pairs using network distance
        # print "Removed %d / %d links based on Euclidean distance" % (len(i) - sum(mask), len(i))
        # i = i[mask]
        # j = j[mask]
        dd_net = (data_target_net.space.getrows(j).distance(data_source_net.space.getrows(i))).toarray(0)
        mask_net = dd_net <= max_d

        link_i.extend(i[mask_net])
        link_j.extend(j[mask_net])

    return np.array(link_i), np.array(link_j)