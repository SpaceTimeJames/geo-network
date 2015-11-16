# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 20:22:58 2013

@author: tobydavies
"""

'''
Version 1

'''

from shapely.geometry import Point, LineString, Polygon
import xml.sax as sax
import pyproj
import pickle as pk
import networkx as nx
from distutils.version import StrictVersion
import numpy as np

from streetnet import StreetNet



class Node():
    
    def __init__(self, feature_id, lonlat, tags):
        self.feature_id = feature_id
        self.lonlat = lonlat
        self.lon, self.lat = lonlat
        self.tags = tags


class Way():
    
    def __init__(self, feature_id, refs, tags):
        self.feature_id = feature_id
        self.nds = refs
        self.tags = tags


class Relation():
    
    def __init__(self,feature_id, members, tags):
        self.feature_id = feature_id
        self.members = members
        self.tags = tags


class OSMHandler(sax.handler.ContentHandler):
    
    def __init__(self):
        sax.handler.ContentHandler.__init__(self)
        self.id = None
        self.geometry = None
        self.tags = None
        self.nodes = {}
        self.ways = {}
        self.relations = {}
    
    def startElement(self, name, attrs):
        if name == 'node':
            self.id = attrs['id']
            self.tags = {}
            self.geometry = (float(attrs['lon']), float(attrs ['lat']))

        elif name == 'way':
            self.id = attrs['id']
            self.tags = {}
            self.geometry = []
        
        elif name == 'nd':
            self.geometry.append(attrs['ref'])
        
        elif name == 'relation':
            self.id = attrs['id']
            self.tags = {}
            self.geometry=[]
        
        elif name == 'member':
            self.geometry.append(attrs['ref'])

        elif name == 'tag':
            self.tags[attrs['k']] = attrs['v']
    
    def endElement(self, name):
        if name == 'node':
            self.nodes[self.id] = Node(self.id, self.geometry, self.tags)
            self.reset()
        elif name == 'way':
            self.ways[self.id] = Way(self.id, self.geometry, self.tags)
            self.reset()
        elif name == 'relation':
            self.relations[self.id] = Relation(self.id, self.geometry, self.tags)
            self.reset()
    
    def reset (self):
        self.id = None
        self.geometry = None
        self.tags = None


class OSMData():
    
    def __init__(self, nodes, ways, relations):
        self.nodes = nodes
        self.ways = ways
        self.relations = relations
    
    def save(self,filename):
        f=open(filename,'wb')
        pk.dump(self,f)
        f.close()


class OSMStreetNet(StreetNet):

    input_srid = 4326
    
    def build_network(self,
                      data,
                      blacklist=('service',)):
        
        self.input_proj = pyproj.Proj(init='epsg:4326')
        self.output_proj = pyproj.Proj(init='epsg:%d' % self.srid)
        
        g_raw = nx.MultiGraph()
        
        highways = dict(
            [(way_id, way) for way_id, way in data.ways.iteritems()
             if 'highway' in way.tags
             and way.tags['highway'] not in blacklist]
        )

        # iterate over ways, adding edges and nodes (nodes are added automatically if they do not exist)
        for way_id, way in highways.iteritems():
            for i in xrange(len(way.nds)-1):
                g_raw.add_edge(way.nds[i], way.nds[i+1])

        # add node locations to the raw network
        for v in g_raw:
            if self.srid is not None:
                g_raw.node[v]['loc'] = pyproj.transform(self.input_proj, self.output_proj, *data.nodes[v].lonlat)
            else:
                g_raw.node[v]['loc'] = data.nodes[v].lonlat

        g = nx.MultiGraph()

        # inline function for adding edges to avoid code repetition
        def add_edge(node_list, edge_count, attrs):
            # create unique key
            key = "%s_%d" % (attrs['fid'], edge_count)
            # define other attributes
            polyline = [g_raw.node[v]['loc'] for v in node_list]
            new_attr = dict(attrs)
            # redefine the edge ID since the original ID is not unique
            new_attr['fid'] = key
            new_attr['linestring'] = LineString(polyline)
            new_attr['length'] = new_attr['linestring'].length
            new_attr['orientation_neg'] = node_list[0]
            new_attr['orientation_pos'] = node_list[-1]

            # close this edge and add to network

            g.add_edge(node_list[0], node_list[-1], key=key, attr_dict=new_attr)
        
        for way_id, way in highways.iteritems():
            
            atts = {
                'fid': way.feature_id,
                'highway': way.tags['highway']
            }
            
            if 'oneway' in way.tags:
                atts['oneway'] = way.tags['oneway']

            if 'junction' in way.tags and way.tags['junction'] == 'roundabout':
                atts['junction'] = 'roundabout'

            # import ipdb; ipdb.set_trace()
            
            # start with the first node in this way
            current_edge_nds = [way.nds[0]]
            edge_count = 1

            # loop over the remaining nodes
            for nd in way.nds[1:]:
                current_edge_nds.append(nd)
                # edge continuation conditions:
                # (1) Current node must have order 2
                # (2) The two nodes connected to current node must be joined by only one edge
                # if these are not fulfilled, end the edge
                if not (len(g_raw[nd]) == 2 and all([len(g_raw[nd][x]) == 1 for x in g_raw[nd]])):
                    add_edge(current_edge_nds, edge_count, atts)
                    edge_count += 1

            # all nodes in this way are exhausted, so check whether there is a final edge to be added
            if len(current_edge_nds) > 1:
                # add final edge
                add_edge(current_edge_nds, edge_count, atts)

        # Only want the largest connected component - sometimes fragments appear
        # round the edge - so take that.
        # NB this interface has changed with NetworkX v1.9
        if StrictVersion(nx.__version__) >= StrictVersion('1.9'):
            try:
                g = sorted(nx.connected_component_subgraphs(g), key=len, reverse=True)[0]
            except IndexError as exc:
                print "Error - probably because the graph is empty"
                raise
        else:
            g = nx.connected_component_subgraphs(g)[0]

        self.g = g

    
    def build_posdict(self, data):
        '''
        Each node gets an attribute added for its geometric position.
        '''
        for node_id in self.g:
            try:
                lon, lat = data.nodes[node_id].lonlat
                if self.srid is not None:
                    x, y = pyproj.transform(self.input_proj, self.output_proj, lon, lat)
                else:
                    x, y = lon, lat
                self.g.node[node_id]['loc'] = (x, y)
            except KeyError:
                import ipdb; ipdb.set_trace()

    
    def build_routing_network(self):
        '''
        Build the routing network
        '''
        
        g_routing=nx.MultiDiGraph()

        #Loop the edges of g and assess the one-way status of each
        for n1,n2,fid,attr in self.g.edges(data=True,keys=True):

            #If one_way attribute is present, only add an edge in the correct direction
            if 'oneway' in attr and attr['oneway'] == 'yes':
                g_routing.add_edge(attr['orientation_neg'],attr['orientation_pos'],key=fid,attr_dict=attr)

            #If the attribute is absent, add edges in both directions
            else:
                g_routing.add_edge(attr['orientation_neg'],attr['orientation_pos'],key=fid,attr_dict=attr)
                g_routing.add_edge(attr['orientation_pos'],attr['orientation_neg'],key=fid,attr_dict=attr)

        for v in g_routing:
            g_routing.node[v]['loc']=self.g.node[v]['loc']

        self.g_routing=g_routing
    
    
#    def plot_network_background(self, bounding_poly=None,
#                                zoom_level=13,
#                                show_edges=True,
#                                show_nodes=False,
#                                edge_width=1,
#                                node_size=7,
#                                edge_col='k',
#                                node_col='r'):
#
#        if bounding_poly is not None:
#            xmin, ymin, xmax, ymax = bounding_poly.bounds
#        else:
#            xmin, ymin, xmax, ymax = self.extent
#
#
#        mm_proj = pyproj.Proj(init='epsg:4326')
#
#        # define transformation function to plot edge patches
#        def transform_fun(x, y):
#            locpt = m.locationPoint(
#                Geo.Location(*pyproj.transform(self.output_proj, mm_proj, x, y)[::-1])
#            )
#            return locpt.x, locpt.y
#
#        xmin, ymin = pyproj.transform(self.output_proj, mm_proj, xmin, ymin)
#        xmax, ymax = pyproj.transform(self.output_proj, mm_proj, xmax, ymax)
#
#        print xmin, ymin, xmax, ymax
#
#        sw = Geo.Location(ymin, xmin)  # lat, lon
#        ne = Geo.Location(ymax, xmax)  # lat, lon
#
#        m = mm.mapByExtentZoom(OpenStreetMap.Provider(), sw, ne, zoom_level)
#
#        lower_left = m.locationPoint(sw)
#        upper_right = m.locationPoint(ne)
#
#        ratio = (lower_left.y-upper_right.y)/(upper_right.x-lower_left.x)
#
#        print 'Aspect ratio '+str(ratio)
#
#        fig=plt.figure(figsize=(8, 8 * ratio))
#        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
#        ax=fig.add_subplot(111)
#
#        if show_edges:
#            for ep in self.generate_edge_patches(transform_fun=transform_fun):
#                ax.add_patch(ep)
#
#        if show_nodes:
#            # NB reverse point order
#            node_coords = [pyproj.transform(self.output_proj, mm_proj, *t[1]['pos'])[::-1] for t in self.g.nodes(data=True)]
#            node_points = [m.locationPoint(Geo.Location(*t)) for t in node_coords]
#            node_points_bbox = [
#                p for p in node_points if lower_left.x <= p.x <= upper_right.x and upper_right.y <= p.y <= lower_left.y
#            ]
#
#            x = [point.x for point in node_points_bbox]
#            y = [point.y for point in node_points_bbox]
#
#            ax.scatter(x, y, c=node_col, s=node_size, zorder=5)
#
#        ax.set_frame_on(False)
#
#        ax.set_xlim(lower_left.x, upper_right.x)
#        ax.set_ylim(lower_left.y, upper_right.y)
#
#        ax.set_xticks([])
#        ax.set_yticks([])
#
#        map_image = m.draw()
#        ax.imshow(map_image)
#
#        
#        # plt.savefig(filename+'.png')
#        # plt.savefig(filename+'.pdf')
#        # plt.savefig(filename+'.eps')
#        #
#        # plt.close('all')


def read_data(filename):
    CurrentHandler = OSMHandler()
    sax.parse(filename, CurrentHandler)
    res = OSMData(CurrentHandler.nodes, CurrentHandler.ways, CurrentHandler.relations)
    return res


if __name__ == '__main__':
    import os
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.path import Path
    this_dir = os.path.dirname(os.path.realpath(__file__))
    OSMFILE = os.path.join(this_dir, 'test_data', 'camden_fragment.osm')
    # OSMFILE = '/Users/tobydavies/osm_data_samples/camden_full.osm'
    osmdata = read_data(OSMFILE)
    o = OSMStreetNet.from_data_structure(osmdata)

    min_x, min_y, max_x, max_y = o.extent
    ratio=(max_y-min_y)/(max_x-min_x)
    
    fig=plt.figure(figsize=(8,8*ratio))
    fig.subplots_adjust(left=0.01,right=0.99,bottom=0.01,top=0.99)
    ax=fig.add_subplot(111)
    
    o.plot_network(ax=ax,show_nodes=True)
    
    
    
    

