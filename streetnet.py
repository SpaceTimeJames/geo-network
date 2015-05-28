__author__ = 'Toby, refactor by Gabs'

from shapely.geometry import Point, LineString, Polygon
import networkx as nx
import cPickle
import scipy as sp
import numpy as np
from collections import defaultdict
import bisect as bs
import pysal as psl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize
from matplotlib.colorbar import ColorbarBase
import copy

class Edge(object):

    def __init__(self, street_net, orientation_pos=None, orientation_neg=None, fid=None, **kwargs):
        self.graph = street_net
        self.orientation_neg = orientation_neg
        self.orientation_pos = orientation_pos
        self.fid = fid

    # redefine __getattr__ so that any dict-style lookups on this object are redirected to look in the attributes
    def __getitem__(self, item):
        return self.attrs[item]

    def __repr__(self):
        return "<Edge {0} <-> {1} ({2})>".format(
            self.orientation_neg,
            self.orientation_pos,
            self.fid
        )

    @property
    def attrs(self):
        return self.graph.g.edge[self.orientation_neg][self.orientation_pos][self.fid]

    @property
    def linestring(self):
        return self.attrs['linestring']

    @property
    def length(self):
        return self.attrs['length']

    @property
    def centroid(self):
        """
        The line's central coordinate as a NetPoint.
        """
        node_dist = {
            self.orientation_neg: self.length / 2.0,
            self.orientation_pos: self.length / 2.0
        }
        return NetPoint(self.graph, self, node_dist)

    @property
    def centroid_xy(self):
        """
        The line's central coordinate in cartesian coordinates.
        """
        return self.centroid.cartesian_coords

    def __eq__(self, other):
        return (
            self.graph is other.graph and
            self.orientation_neg == other.orientation_neg and
            self.orientation_pos == other.orientation_pos and
            self.fid == other.fid
        )


class NetPoint(object):

    def __init__(self, street_net, edge, node_dist):
        """
        :param street_net: A pointer to the network on which this point is defined
        :param edge: An edge ID referring to an edge in street_net
        :param node_dist: A dictionary containing the distance along this edge from both the positive and negative end
        The key gives the node ID, the value gives the distance from that end.
        """
        self.graph = street_net
        self.edge = edge
        self.node_dist = node_dist

    @property
    def cartesian_coords(self):
        return self.graph.network_point_to_xy(self)

    def test_compatible(self, other):
        if not self.graph is other.graph:
            raise AttributeError("The two points are defined on different graphs")

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            raise TypeError("Can only compare NetPoint with another NetPoint.")
        # don't use test_compatible here because we want such an operation to return False, not raise an exception
        return (
            self.graph is other.graph and
            self.edge == other.edge and
            self.node_dist == other.node_dist
        )

    def __sub__(self, other):
        # NetPoint - NetPoint -> NetPath
        self.test_compatible(other)
        try:
            if self.graph.directed:
                return self.graph.path_directed(self, other)
            else:
                return self.graph.path_undirected(self, other)
        except ValueError:
            import ipdb; ipdb.set_trace()


class NetPath(object):

    def __init__(self, start, end, edges, distances, nodes):

        self.start = start
        self.end = end
        self.edges = edges
        self.distances = distances
        self.nodes = nodes

        if len(distances) != len(edges):
            raise AttributeError('Path mismatch: distance list wrong length')

        if len(nodes) != len(edges) - 1:
            raise AttributeError('Path mismatch: node list wrong length')

        if self.start.graph is not self.end.graph:
            raise AttributeError('Path mismatch: nodes are defined on different graphs')

        self.graph = self.start.graph

    @property
    def length(self):
        return sum(self.distances)

    @property
    def node_degrees(self):
        return [self.graph.g.degree(t) for t in self.nodes]


class GridEdgeIndex(object):

    def __init__(self, x_grid, y_grid, edge_index):

        self.x_grid = x_grid
        self.y_grid = y_grid
        self.edge_index = edge_index


class StreetNet(object):

    '''
    Main street network base (virtual) class.

    It is always initialised empty, and can then be provided data in two ways: by
    building a network from an ITNData instance or similar, or by inheriting an already-built
    network from elsewhere.

    The reason for the latter is cleaning - sometimes it is useful to produce a
    'new' StreetNet derived from a previous one.

    The graph used is a multigraph - sometimes multiple edges exist between nodes
    (a lay-by is the most obvious example). That causes a fair bit of pain, but
    there is no alternative.

    It also builds an additional 'routing graph', which is an exact copy of the physical
    graph, except directed with edges present corresponding to whether travel is
    allowed. This is built by the method build_routing_network(), which is run
    automatically when either loading from data or inheriting.

    The graph is undirected - defer dealing with routing information for now, although
    often undirected is preferable anyway.

    NetworkX networks are stored using a dictionary-of-dictionaries model. Each edge
    is technically a triple - the two terminal nodes and a set of attributes. There
    are loads of ways of accessing and manipulating these and iterating over them.

    All cleaning routines stripped out for now because they're uncommented, ugly
    and bloated.
    '''

    def __init__(self, routing='undirected'):
        '''
        This just initialises a fresh empty network in each new class. Will be
        overwritten but just ensures that stuff won't break.
        :param routing: Defines the behaviour upon subtracting two NetPoints
        '''

        self.g = nx.MultiGraph()
        self.g_routing = nx.MultiDiGraph()
        self.directed = routing.lower() == 'directed'


    @classmethod
    def from_data_structure(cls, data):
        obj = cls()
        print 'Building the network'
        obj.build_network(data)

        print 'Building position dictionary'
        obj.build_posdict(data)

        print 'Building routing network'
        obj.build_routing_network()

        return obj

    @classmethod
    def from_pickle(cls, filename):
        with open(filename, 'r') as f:
            obj = cPickle.load(f)
        return obj

    @classmethod
    def from_multigraph(cls, g):
        obj = cls()
        obj.g = g
        obj.build_routing_network()
        return obj

    def build_network(self, data):

        raise NotImplementedError()


    def build_posdict(self, data):
        '''
        Each node gets an attribute added for its geometric position. This is only
        really useful for plotting.
        '''
        raise NotImplementedError()


    def build_routing_network(self):
        '''
        This builds a second graph for the class representing the routing information.

        It is a directed multigraph in which direction represents allowed travel.
        Edges are present in both directions if travel is two-way, otherwise only
        one is included.

        All other attributes are exactly the same as the underlying g, and inherited
        as such.
        '''
        raise NotImplementedError()


    def plot_network(self,
                     ax=None,
                     extent=None,
                     show_edges=True,
                     show_nodes=False,
                     edge_width=1,
                     node_size=7,
                     edge_outer_col='k',
                     edge_inner_col=None,
                     node_col='r'):

        '''
        This plots the section of the network that lies within a given bounding box.
        Updated to use a PatchCollection, which is faster to plot.
        :param ax: Optional axis handle for plotting, otherwise use the current axes/make a new figure.
        :param edge_inner_col: [Optional] If a scalar, fill all edges with this value.
        If a dict then each key is an edge ID with corresponding value indicating the fill colour for that edge.
        '''
        min_x, min_y, max_x, max_y = extent if extent is not None else self.extent
        bounding_poly = Polygon((
            (min_x, min_y),
            (max_x, min_y),
            (max_x, max_y),
            (min_x, max_y),
            (min_x, min_y)
        ))
        ax = ax if ax is not None else plt.gca()
        if show_edges:

            path_patches = []

            for e in self.edges():
                fid = e['fid']
                ls = e['linestring']
            # for n1,n2,fid,attr in self.g.edges(data=True, keys=True):

                # bbox_check=[min_x<=p[0]<=max_x and min_y<=p[1]<=max_y for p in attr['polyline']]
                bbox_check = bounding_poly.intersects(ls)

                #This checks that at least some of the line lies within the bounding
                #box. This is to avoid creating unnecessary lines which will not
                #actually be seen.
                # if any(bbox_check):
                if bbox_check:

                    path = Path(ls)
                    path_patches.append(patches.PathPatch(path, facecolor='none', edgecolor=edge_outer_col, lw=edge_width))
                    # patch = patches.PathPatch(path, facecolor='none', edgecolor=edge_outer_col, lw=edge_width)
                    # ax.add_patch(patch)

                    if edge_inner_col is not None:
                        if isinstance(edge_inner_col, dict):
                            ec = edge_inner_col.get(fid, 'w')
                        else:
                            ec = edge_inner_col
                        # patch = patches.PathPatch(path, facecolor='none', edgecolor=ec, lw=0.6*edge_width, zorder=2)
                        # ax.add_patch(patch)
                        path_patches.append(patches.PathPatch(path, facecolor='none', edgecolor=ec, lw=0.6*edge_width, zorder=2))

                    #These circles are a massive fudge to give the lines 'rounded'
                    #ends. They look nice, but best to only do them at the last
                    #minute because it is hard to work out what radius they should
                    #be - the units scale differently to edge_width so need to be
                    #trial-and-errored each time.
                    #TODO: Calculate these radii automatically
#                    end1=patches.Circle(attr['polyline'][0],radius=3.2*edge_width,facecolor='k',edgecolor='k',lw=0,zorder=1)
#                    ax.add_patch(end1)
#                    end2=patches.Circle(attr['polyline'][-1],radius=3.2*edge_width,facecolor='k',edgecolor='k',lw=0,zorder=1)
#                    ax.add_patch(end2)

            ax.add_collection(PatchCollection(path_patches, match_original=True))

        if show_nodes:
            node_points=[self.g.node[v]['loc'] for v in self.g]
            node_points_bbox=[p for p in node_points if min_x<=p[0]<=max_x and min_y<=p[1]<=max_y]
            x,y = zip(*node_points_bbox)
            ax.scatter(x,y,c=node_col,s=node_size,zorder=5)

        ax.set_xlim(min_x,max_x)
        ax.set_ylim(min_y,max_y)
        ax.set_aspect('equal')

    def within_boundary(self, poly, outer_buffer=0):

        '''
        This cuts out only the part of the network which lies within some specified
        polygon (e.g. the boundary of a borough). The polygon is passed as a chain
        of values, and then a lot of the work is done by Shapely.

        A buffer can also be passed - this enlarges the boundary in the obvious way.

        This is an example of the 'inheritance' method - a whole new network is
        produced, and the output of the routine is a new instance of ITNStreetNet.
        '''

        #Create new graph
        g_new=nx.MultiGraph()

        #Make a shapely polygon from the boundary
        boundary=Polygon(poly)

        #Buffer it
        boundary=boundary.buffer(outer_buffer)

        #Loop the edges
        for n1,n2,fid,attr in self.g.edges(data=True,keys=True):
            #Make a shapely polyline for each
            # edge_line=LineString(attr['polyline'])
            edge_line = attr['linestring']

            #Check intersection
            if edge_line.intersects(boundary):
                #Add edge to new graph
                g_new.add_edge(n1,n2,key=fid,attr_dict=attr)

        #Add all nodes to the new posdict
        for v in g_new:
            g_new.node[v]['loc']=self.g.node[v]['loc']

        # generate a new object from this multigraph
        return self.__class__.from_multigraph(g_new)

    def save(self, filename):
        '''
        Just saves the network for convenience.
        '''
        with open(filename, 'wb') as f:
            cPickle.dump(self, f)


    def build_grid_edge_index(self, gridsize, extent=None):
        '''
        This is a helper function really, to be used in conjunction with snapping
        operations. This might be a completely stupid way to do it, but as a naive
        method it works OK.

        The rationale is: suppose you want to snap a point to the closest edge -
        then in the first instance you hav to go through every edge, calculate
        the distance to it, then find the minimum. This is really inefficient because
        some of those edges are miles away and each distance calculation is expensive.

        The idea here is to lay a grid across the space and, for each cell of the
        grid, take a note of which edges intersect with it. Then, as long as the grid
        is suitably-sized, snapping a point is much cheaper - you find which cell
        it lies in, and then you only need calculate the distance to the edges which
        pass through any of the neighbouring cells - closest edge will always be in
        one of those.

        There *must* be a better way, but this works in the meantime.

        This constructs the grid cells, then it constructs a dictionary edge-locator:
        for each index pair (i,j), corresponding to the cells, it gives a list of
        all edges which intersect that cell.

        In fact, because it is just done by brute force, it doesn't check for intersection,
        rather it adds every edge to all cells that it *might* intersect. Lazy, could
        definitely be optimised, but still produces huge saving even like this.

        The reason for pre-computing this is that it may be necessary to use it
        many times as the argument to other functions.
        '''
        min_x, min_y, max_x, max_y = extent or self.extent

        #Set up grids and arrays
        x_grid=sp.arange(min_x,max_x,gridsize)
        y_grid=sp.arange(min_y,max_y,gridsize)

        #Initialise the lookup
        edge_index=defaultdict(list)

        #Loop edges
        for n1, n2, fid, attr in self.g.edges(data=True, keys=True):

            edge_line = attr['linestring']

            #Get bounding box of polyline
            (bbox_min_x, bbox_min_y, bbox_max_x, bbox_max_y) = edge_line.bounds

            #Bin bbox extremities
            bbox_min_x_loc = bs.bisect_left(x_grid, bbox_min_x)
            bbox_max_x_loc = bs.bisect_left(x_grid, bbox_max_x)
            bbox_min_y_loc = bs.bisect_left(y_grid, bbox_min_y)
            bbox_max_y_loc = bs.bisect_left(y_grid, bbox_max_y)

            #Go through every cell covered by this bbox, augmenting lookup
            for i in range(bbox_min_x_loc,bbox_max_x_loc+1):
                for j in range(bbox_min_y_loc,bbox_max_y_loc+1):
                    edge_index[(i,j)].append((n1,n2,fid))

        grid_edge_index=GridEdgeIndex(x_grid,y_grid,edge_index)

        return grid_edge_index


    def closest_edges_euclidean(self,x,y,grid_edge_index,radius=50,max_edges=1):
        '''
        Snap a point to the closest segment.

        Needs to be passed the output from a previous run of bin_edges().

        The radius argument allows an upper limit on the snap distance to be imposed -
        'if there are no streets within radius metres, return empty list'. This is
        useful in some circumstances.
        '''
        #TODO: Raise warning if grid size smaller than radius

        #Produce a shapely point and find which cell it lies in.
        point = Point(x,y)

        x_loc = bs.bisect_left(grid_edge_index.x_grid, x)
        y_loc = bs.bisect_left(grid_edge_index.y_grid, y)

        #Go round this cell and all neighbours (9 in total) collecting all edges
        #which could intersect with any of these.
        candidate_edges=[]

        for i in [-1,0,1]:
            for j in [-1,0,1]:
                for e in grid_edge_index.edge_index[(x_loc+i,y_loc+j)]:
                    if e not in candidate_edges:
                        candidate_edges.append(e)

        #candidate_edges now contains all candidates for closest edge

        #Calculate the distances to each
        # candidate_edge_distances=[point.distance(LineString(self.g[n1][n2][fid]['polyline'])) for (n1,n2,fid) in candidate_edges]
        candidate_edge_distances=[point.distance(self.g[n1][n2][fid]['linestring']) for (n1, n2, fid) in candidate_edges]

        #Order the edges according to proximity, omitting those which are further than radius away
        valid_edges_distances=[w for w in sorted(zip(candidate_edges,candidate_edge_distances),key=lambda w: w[1]) if w[1]<radius]

        closest_edges=[]

        for (n1,n2,fid),snap_distance in valid_edges_distances[:max_edges]:

            #Do various proximity calculations

            # polyline=LineString(self.g[n1][n2][fid]['polyline'])
            polyline = self.g[n1][n2][fid]['linestring']

            #node_dist is a lookup, indexed by each of the terminal nodes of closest_edge,
            #which gives the distance from that node to the point on the line to which
            #the original point snaps to. If that makes sense.
            #The polyline is specified from negative to positive orientation BTW.
            node_dist={}

            node_dist[self.g[n1][n2][fid]['orientation_neg']]=polyline.project(point)
            node_dist[self.g[n1][n2][fid]['orientation_pos']]=polyline.length-polyline.project(point)

            edge = Edge(self, **self.g[n1][n2][fid])
            net_point = NetPoint(self, edge, node_dist)

            closest_edges.append((net_point, snap_distance))

        return closest_edges


    def path_undirected(self,net_point1,net_point2):

        n1_1 = net_point1.edge.orientation_neg
        n2_1 = net_point1.edge.orientation_pos
        fid_1 = net_point1.edge.fid

        n1_2 = net_point2.edge.orientation_neg
        n2_2 = net_point2.edge.orientation_pos
        fid_2 = net_point2.edge.fid

        # n1_1,n2_1,fid_1=net_point1.edge
        # n1_2,n2_2,fid_2=net_point2.edge

        node_dist1=net_point1.node_dist
        node_dist2=net_point2.node_dist

        if net_point1.edge == net_point2.edge:  # both points on same edge

            dist_diff = node_dist2[n1_1] - node_dist1[n1_1]

            path_edges=[fid_1]
            path_distances=[abs(dist_diff)]
            path_nodes=[]

            path=NetPath(net_point1,net_point2,path_edges,path_distances,path_nodes)

        else:

            #Insert a fresh pair of nodes
            self.g.add_edge(n1_1,'point1',key=fid_1,length=node_dist1[n1_1])
            self.g.add_edge('point1',n2_1,key=fid_1,length=node_dist1[n2_1])

            #As above, store edge data before removal
            removed_edge1_atts=self.g[n1_1][n2_1][fid_1]

            self.g.remove_edge(n1_1,n2_1,fid_1)



            #Insert a fresh pair of nodes
            self.g.add_edge(n1_2,'point2',key=fid_2,length=node_dist2[n1_2])
            self.g.add_edge('point2',n2_2,key=fid_2,length=node_dist2[n2_2])

            #As above, store edge data before removal
            removed_edge2_atts=self.g[n1_2][n2_2][fid_2]

            self.g.remove_edge(n1_2,n2_2,fid_2)


            #Get the path between the new nodes
            #TODO: CONSIDER BIDIRECTIONAL DIJKSTRA
            try:

                distances, paths = nx.single_source_dijkstra(self.g,'point1',target='point2',weight='length')

                node_path=paths['point2']

                path_edges=[]
                path_distances=[]
                path_nodes=node_path[1:-1]

                for j in xrange(len(node_path)-1):

                    v=node_path[j]
                    w=node_path[j+1]

                    fid_shortest=min(self.g[v][w],key=lambda x: self.g[v][w][x]['length'])

                    path_edges.append(fid_shortest)
                    path_distances.append(self.g[v][w][fid_shortest]['length'])

                path=NetPath(net_point1,net_point2,path_edges,path_distances,path_nodes)

            except:

                path=None


            #Restore the network to original state
            self.g.remove_node('point1')
            self.g.remove_node('point2')

            self.g.add_edge(n1_1,n2_1,key=fid_1,attr_dict=removed_edge1_atts)
            self.g.add_edge(n1_2,n2_2,key=fid_2,attr_dict=removed_edge2_atts)

        return path


    def path_directed(self, net_point1, net_point2):

        n1_1 = net_point1.edge.orientation_neg
        n2_1 = net_point1.edge.orientation_pos
        fid_1 = net_point1.edge.fid

        n1_2 = net_point2.edge.orientation_neg
        n2_2 = net_point2.edge.orientation_pos
        fid_2 = net_point2.edge.fid

        # n1_1,n2_1,fid_1=net_point1.edge
        # n1_2,n2_2,fid_2=net_point2.edge

        node_dist1=net_point1.node_dist
        node_dist2=net_point2.node_dist

        if fid_1==fid_2:

            dist_diff=node_dist2[n1_1]-node_dist1[n1_1]

            if dist_diff==0:

                path_edges=[fid_1]
                path_distances=[dist_diff]
                path_nodes=[]

                path=NetPath(net_point1,net_point2,path_edges,path_distances,path_nodes)

            else:

                #p1_node is the node for which p1 is the closer of the two points

                if dist_diff>0:

                    p1_node=n1_1
                    p2_node=n2_1

                else:

                    p1_node=n2_1
                    p2_node=n1_1

                if p2_node in self.g_routing[p1_node]:

                    path_edges=[fid_1]
                    path_distances=[node_dist2[p1_node]-node_dist1[p1_node]]
                    path_nodes=[]

                    path=NetPath(net_point1,net_point2,path_edges,path_distances,path_nodes)

                else:

                    try:

                        distances, paths = nx.single_source_dijkstra(self.g_routing,p1_node,target=p2_node,weight='length')

                        node_path=paths[p2_node]

                        path_edges=[fid_1]
                        path_distances=[node_dist1[p1_node]]
                        path_nodes=[p1_node]

                        for j in xrange(len(node_path)-1):

                            v=node_path[j]
                            w=node_path[j+1]

                            fid_shortest=min(self.g_routing[v][w],key=lambda x: self.g_routing[v][w][x]['length'])

                            path_edges.append(fid_shortest)
                            path_distances.append(self.g_routing[v][w][fid_shortest]['length'])
                            path_nodes.append(w)

                        path_edges.append(fid_1)
                        path_distances.append(node_dist2[p2_node])

                        path=NetPath(net_point1,net_point2,path_edges,path_distances,path_nodes)

                    except:

                        path=None


        else:

            removed_edges1=[]
            removed_edges1_atts=[]

            if n2_1 in self.g_routing[n1_1]:
                if fid_1 in self.g_routing[n1_1][n2_1]:

                    self.g_routing.add_edge(n1_1,'point1',key=fid_1,length=node_dist1[n1_1])
                    self.g_routing.add_edge('point1',n2_1,key=fid_1,length=node_dist1[n2_1])

                    removed_edge1=(n1_1,n2_1,fid_1)
                    removed_edge1_atts=self.g_routing[n1_1][n2_1][fid_1]

                    removed_edges1.append(removed_edge1)
                    removed_edges1_atts.append(removed_edge1_atts)

                    self.g_routing.remove_edge(n1_1,n2_1,fid_1)

            if n1_1 in self.g_routing[n2_1]:
                if fid_1 in self.g_routing[n2_1][n1_1]:

                    self.g_routing.add_edge(n2_1,'point1',key=fid_1,length=node_dist1[n2_1])
                    self.g_routing.add_edge('point1',n1_1,key=fid_1,length=node_dist1[n1_1])

                    removed_edge1=(n2_1,n1_1,fid_1)
                    removed_edge1_atts=self.g_routing[n2_1][n1_1][fid_1]

                    removed_edges1.append(removed_edge1)
                    removed_edges1_atts.append(removed_edge1_atts)

                    self.g_routing.remove_edge(n2_1,n1_1,fid_1)



            removed_edges2=[]
            removed_edges2_atts=[]

            if n2_2 in self.g_routing[n1_2]:
                if fid_2 in self.g_routing[n1_2][n2_2]:

                    self.g_routing.add_edge(n1_2,'point2',key=fid_2,length=node_dist2[n1_2])
                    self.g_routing.add_edge('point2',n2_2,key=fid_2,length=node_dist2[n2_2])

                    removed_edge2=(n1_2,n2_2,fid_2)
                    removed_edge2_atts=self.g_routing[n1_2][n2_2][fid_2]

                    removed_edges2.append(removed_edge2)
                    removed_edges2_atts.append(removed_edge2_atts)

                    self.g_routing.remove_edge(n1_2,n2_2,fid_2)

            if n1_2 in self.g_routing[n2_2]:
                if fid_2 in self.g_routing[n2_2][n1_2]:

                    self.g_routing.add_edge(n2_2,'point2',key=fid_2,length=node_dist2[n2_2])
                    self.g_routing.add_edge('point2',n1_2,key=fid_2,length=node_dist2[n1_2])

                    removed_edge2=(n2_2,n1_2,fid_2)
                    removed_edge2_atts=self.g_routing[n2_2][n1_2][fid_2]

                    removed_edges2.append(removed_edge2)
                    removed_edges2_atts.append(removed_edge2_atts)

                    self.g_routing.remove_edge(n2_2,n1_2,fid_2)


            #Get the path between the new nodes
            #TODO: CONSIDER BIDIRECTIONAL DIJKSTRA
            try:

                distances, paths = nx.single_source_dijkstra(self.g_routing,'point1',target='point2',weight='length')

                node_path=paths['point2']

                path_edges=[]
                path_distances=[]
                path_nodes=node_path[1:-1]

                for j in xrange(len(node_path)-1):

                    v=node_path[j]
                    w=node_path[j+1]

                    fid_shortest=min(self.g_routing[v][w],key=lambda x: self.g_routing[v][w][x]['length'])

                    path_edges.append(fid_shortest)
                    path_distances.append(self.g_routing[v][w][fid_shortest]['length'])

                path=NetPath(net_point1,net_point2,path_edges,path_distances,path_nodes)

            except:

                path=None


            #Restore the network to original state
            self.g_routing.remove_node('point1')
            self.g_routing.remove_node('point2')

            for (n1,n2,fid),removed_edge_atts in zip(removed_edges1,removed_edges1_atts):
                self.g_routing.add_edge(n1,n2,key=fid,attr_dict=removed_edge_atts)

            for (n1,n2,fid),removed_edge_atts in zip(removed_edges2,removed_edges2_atts):
                self.g_routing.add_edge(n1,n2,key=fid,attr_dict=removed_edge_atts)

        return path

    ### ADDED BY GABS
    def edges(self, bounding_poly=None):
        '''
        Get all edges in the network.  Optionally return only those that intersect the provided bounding polygon
        '''
        if bounding_poly:
            return [Edge(self, **x[2]) for x in self.g.edges(data=True) if bounding_poly.intersects(x[2]['linestring'])]
        else:
            return [Edge(self, **x[2]) for x in self.g.edges(data=True)]

    ### ADDED BY GABS
    def nodes(self, bounding_poly=None):
        """
        Get all nodes in the network. Optionally return only those that intersect the provided bounding polygon
        """
        if bounding_poly:
            return [x[0] for x in self.g.nodes(data=True) if bounding_poly.intersects(Point(*x[1]['loc']))]
        else:
            return self.g.nodes()

    ### ADDED BY GABS
    def edge(self, node1, node2, fid):
        return self.g.edge[node1, node2, fid]

    ### ADDED BY GABS
    def lines_iter(self):
        """
        Returns a generator that iterates over all edge linestrings.
        This is useful for various spatial operations.
        """
        for e in self.g.edges_iter(data=True):
            yield e[2]['linestring']

    ### ADDED BY GABS
    def closest_segments_euclidean_brute_force(self, x, y, radius=None):
        pt = Point(x, y)
        if radius:
            bpoly = pt.buffer(radius)
        else:
            bpoly = None

        edges = self.edges(bounding_poly=bpoly)
        if not len(edges):
            # no valid edges found, bail.
            return None

        snap_distances = [x.distance(pt) for x in self.lines_iter()]
        idx = np.argmin(snap_distances)
        snap_distance = snap_distances[idx]
        closest_edge = edges[idx]

        da = closest_edge['linestring'].project(pt)
        dist_along = {
            closest_edge.orientation_neg: da,
            closest_edge.orientation_pos: closest_edge['linestring'].length - da,
        }

        return NetPoint(self, closest_edge, dist_along), snap_distance

    ### ADDED BY GABS
    @property
    def extent(self):
        """
        Compute the rectangular bounding coordinates of the edges
        """
        xmin = np.inf
        ymin = np.inf
        xmax = -np.inf
        ymax = -np.inf

        for l in self.lines_iter():
            a, b, c, d = l.bounds
            xmin = min(xmin, a)
            ymin = min(ymin, b)
            xmax = max(xmax, c)
            ymax = max(ymax, d)

        return xmin, ymin, xmax, ymax

    ### ADDED BY GABS
    def network_point_to_xy(self, net_point):
        """
        Convert a NetPoint to cartesian coordinates. This uses linear interpolation along the street segment.
        Assumes that the linestring is defined from negative to positive end
        :return: (x, y) coordinates corresponding to supplied point
        """
        ls = net_point.edge['linestring']
        pt = ls.interpolate(net_point.node_dist[net_point.edge.orientation_neg])
        return pt.x, pt.y
