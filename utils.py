__author__ = 'gabriel'
from point_process.utils import pairwise_differences_indices
import numpy as np


def network_linkages(data_source, max_t, max_d, data_target=None, chunksize=2**18, remove_coincident_pairs=False):
    ###TODO: update this for network distances - shouldn't be too difficult
    """
    Compute the indices of datapoints that are within the following tolerances:
    interpoint distance less than max_d
    time difference greater than zero, less than max_t
    The sign convention is (target - source).  Distances are euclidean.
    :param data_source: EuclideanSpaceTimeData array of source data.  Must be sorted by time ascending.
    :param max_t: maximum time difference (minimum is always zero)
    :param max_d: maximum spatial distance
    :param data_target: optional EuclideanSpaceTimeData array.  If supplied, the linkage indices are between
    data_source and data_target, otherwise the two are set equal
    :param chunksize: The size of an iteration chunk.
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
        dt = (data_target.time.getrows(j) - data_source.time.getrows(i)).toarray(0)
        dd = (data_target.space.getrows(j).distance(data_source.space.getrows(i))).toarray(0)
        mask = (dt <= max_t) & (dt > 0.) & (dd <= max_d)
        if remove_coincident_pairs:
            mask = mask & (dd != 0)
        link_i.extend(i[mask])
        link_j.extend(j[mask])

    return np.array(link_i), np.array(link_j)