__author__ = 'gabriel'
from validation import hotspot
from data.models import NetworkSpaceTimeData, NetworkData, DataArray
from point_process.utils import linkages
from scipy import sparse
import numpy as np

class NetworkProMap(hotspot.STKernelBase):

    data_class = NetworkSpaceTimeData

    def __init__(self, a, b):
        self.a = a
        self.b = b
        super(NetworkProMap, self).__init__()

    def predict(self, data_array):
        data_array = self.data_class(data_array)

        # construct linkage indices
        # e = 0.005
        e = 1e-6
        max_delta_t = (1 - e) / (self.a * e)
        max_delta_d = (1 - e) / (self.b * e)

        link_i, link_j = linkages(self.data, max_delta_t, max_delta_d, data_target=data_array)

        m = sparse.lil_matrix((self.data.ndata, data_array.ndata))
        tt = 1 / ((data_array.time.getrows(link_j) - self.data.time.getrows(link_i)) * self.a + 1.).toarray(0)
        dd = 1 / (data_array.space.getrows(link_j).distance(self.data.space.getrows(link_i)) * self.b + 1.).toarray(0)

        m[link_i, link_j] = tt * dd
        m[tt < 0] = 0.

        return np.array(m.sum(axis=0).flat)
