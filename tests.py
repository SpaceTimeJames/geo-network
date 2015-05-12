__author__ = 'gabriel'
from network.itn import read_gml, ITNStreetNet
import os
import settings

IN_FILE = os.path.join(settings.DATA_DIR, 'network_data', 'mastermap-itn_417209_0_brixton_sample.gml')

# TestData=read_gml('../network_data/mastermap-itn_417209_0_brixton_sample.gml')
#TestData=read_gml('../network_data/mastermap-itn_544003_0_camden_buff2000.gml')
test_data = read_gml(IN_FILE)

itn_net = ITNStreetNet()
itn_net.load_from_data(test_data)

grid_edge_index=itn_net.build_grid_edge_index(530850, 531900, 174550, 175500, 50)


#Four test points - 1 and 3 on same segment, 2 on neighbouring segment, 4 long way away.
#5 and 6 are created so that there are 2 paths of almost-equal length between them - they
#lie on opposite sides of a 'square'
net_point1, snap_dist = itn_net.closest_edges_euclidean(531190, 175214, grid_edge_index)[0]
net_point2, snap_dist = itn_net.closest_edges_euclidean(531149, 175185, grid_edge_index)[0]
net_point3, snap_dist = itn_net.closest_edges_euclidean(531210, 175214, grid_edge_index)[0]
net_point4, snap_dist = itn_net.closest_edges_euclidean(531198, 174962, grid_edge_index)[0]
net_point5, snap_dist = itn_net.closest_edges_euclidean(531090, 175180, grid_edge_index)[0]
net_point6, snap_dist = itn_net.closest_edges_euclidean(531110, 175110, grid_edge_index)[0]
net_point7, snap_dist = itn_net.closest_edges_euclidean(531050, 175300, grid_edge_index)[0]
net_point8, snap_dist = itn_net.closest_edges_euclidean(530973, 175210, grid_edge_index)[0]
net_point9, snap_dist = itn_net.closest_edges_euclidean(530975, 175217, grid_edge_index)[0]

