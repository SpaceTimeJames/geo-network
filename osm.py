# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 20:22:58 2013

@author: tobydavies
"""

'''
Version 1

TODO:
Metric calculation
Segment colour plotting
Inheritance via boundary
'''

from shapely.geometry import Point, LineString, Polygon
import xml.sax as sax
import pyproj
import pickle as pk
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
from distutils.version import StrictVersion
import ModestMaps as mm
from ModestMaps import Geo
from ModestMaps import OpenStreetMap
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
    
    def startElement(self,name,attrs):
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
    
    def endElement(self,name):
        if name == 'node':
            self.nodes[self.id] = Node(self.id, self.geometry, self.tags)
            self.reset()
        elif name == 'way':
            self.ways[self.id] = Way(self.id, self.geometry, self.tags)
            self.reset()
        elif name == 'relation':
            self.relations[self.id] = Relation(self.id,self.geometry,self.tags)
            self.reset()
    
    def reset (self):
        self.id = None
        self.geometry = None
        self.tags = None


class OSMData():
    
    def __init__(self,nodes,ways,relations):
        self.nodes = nodes
        self.ways = ways
        self.relations = relations
    
    def save(self,filename):
        f=open(filename,'wb')
        pk.dump(self,f)
        f.close()


class OSMStreetNet(StreetNet):

    input_srid = 4326
    srid = 27700
    
    
    def build_network(self,data):
        
        self.input_proj = pyproj.Proj(init='epsg:4326')
        self.output_proj = pyproj.Proj(init='epsg:%d' % self.srid)
        
        g_raw = nx.MultiGraph()
        
        highways = [way for way in data.ways.values() if 'highway' in way.tags]
        
        blacklist=['footway', 'service']
        
        highways = [way for way in highways if way.tags['highway'] not in blacklist]
        
        for way in highways:

            for i in xrange(len(way.nds)-1):
                g_raw.add_edge(way.nds[i], way.nds[i+1])
        
        for v in g_raw:
            if self.srid is not None:
                g_raw.node[v]['loc']=pyproj.transform(self.input_proj, self.output_proj, *data.nodes[v].lonlat)
            else:
                g_raw.node[v]['loc']=data.nodes[v].lonlat
            
        g = nx.MultiGraph()
        
        for way in highways:
            
            atts = {
                'fid': way.feature_id,
                'highway': way.tags['highway']
            }

            if 'junction' in way.tags and way.tags['junction'] == 'roundabout':
                atts['junction'] = 'roundabout'
            
            current_edge_nds=[way.nds[0]]
            
            for nd in way.nds[1:]:
                current_edge_nds.append(nd)
                if not (len(g_raw[nd])==2 and all([len(g_raw[nd][x])==1 for x in g_raw[nd]])):
                    polyline = [g_raw.node[v]['loc'] for v in current_edge_nds]
                    atts['linestring'] = LineString(polyline)
                    atts['length'] = atts['linestring'].length
                    atts['orientation_neg']=current_edge_nds[0]
                    atts['orientation_pos']=current_edge_nds[-1]
                    
                    g.add_edge(current_edge_nds[0],current_edge_nds[-1],attr_dict=atts)
                    
                    current_edge_nds=[nd]
                
            if len(current_edge_nds)>1:
                
                polyline = [g_raw.node[v]['loc'] for v in current_edge_nds]
                atts['linestring'] = LineString(polyline)
                atts['length'] = atts['linestring'].length
                atts['orientation_neg']=current_edge_nds[0]
                atts['orientation_pos']=current_edge_nds[-1]
                
                g.add_edge(current_edge_nds[0],current_edge_nds[-1],attr_dict=atts)
                
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
        pass
    
    
#    def within_boundary(self, poly):
#        # poly is a Shapely polygon defining the boundary
#        # NB the coordinates must be in the same projection as this network - they are NOT checked
#
#        #Create new graph
#        g_new=nx.MultiGraph()
#
#        nodes_to_add = []
#        edges_to_add = []
#        
#        #Loop the edges
#        for e in self.g.edges(data=True):
#            if e[2]['linestring'].intersects(poly):
#                # add nodes and this edge - coordinates are attached to the nodes in the attribute dictionary
#                edges_to_add.append(e)
#                nodes_to_add.append((e[0], self.g.node[e[0]]))
#                nodes_to_add.append((e[1], self.g.node[e[1]]))
#
#        g_new.add_nodes_from(nodes_to_add)
#        g_new.add_edges_from(edges_to_add)
#
#        net_new=OSMStreetNet()
#        net_new.inherit(g_new)
#        
#        return net_new
#
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
#
#    def lines_iter(self):
#        """
#        Returns a generator that iterates over all edge linestrings.
#        This is useful for various spatial operations.
#        """
#        if not len(self.g):
#            raise AttributeError("Graph is empty")
#
#        for e in self.g.edges_iter(data=True):
#            yield e[2]['linestring']
#
#    @property
#    def extent(self):
#        """
#        Compute the rectangular bounding coordinates of the edges
#        """
#        xmin = np.inf
#        ymin = np.inf
#        xmax = -np.inf
#        ymax = -np.inf
#
#        for l in self.lines_iter():
#            a, b, c, d = l.bounds
#            xmin = min(xmin, a)
#            ymin = min(ymin, b)
#            xmax = max(xmax, c)
#            ymax = max(ymax, d)
#
#        return xmin, ymin, xmax, ymax
#
#    def edges(self, bpoly=None):
#        '''
#        Get all edges in the network.  Optionally return only those that intersect the provided bounding polygon
#        '''
#        if bpoly:
#            return [x for x in self.g.edges(data=True) if bpoly.intersects(x[2]['linestring'])]
#        else:
#            return self.g.edges(data=True)
#
#    def generate_edge_patches(self, bbox=None, transform_fun=None, **kwargs):
#        '''
#        Iteratively generate path patches for every visible edge
#
#        :param bbox: Optional shapely Polygon defining the visible area.  If not supplied, generate ALL edges
#        :param transform_fun: Optional projection function.  Function takes a pair of coords (x, y) and returns the
#        transformed pair.  If supplied, transform coords before creating patch.
#        :param kwargs: dictionary of options passed to PathPatch
#        :return: array of type PathPatch for adding to a matplotlib axis.  Optionally, can add edge caps too
#        '''
#
#        res = []
#        for ls in self.lines_iter():
#                #This checks that at least some of the line lies within the bounding
#                #box. This is to avoid creating unnecessary lines which will not
#                #actually be seen.
#                if bbox is None or ls.intersects(bbox):
#                    if transform_fun:
#                        ls = LineString([transform_fun(x, y) for x, y in zip(
#                            ls.xy[0].tolist(),
#                            ls.xy[1].tolist()
#                        )])
#
#                    path = Path(ls)
#                    res.append(patches.PathPatch(path, facecolor='none', **kwargs))
#
#                    #These circles are a massive fudge to give the lines 'rounded'
#                    #ends. They look nice, but best to only do them at the last
#                    #minute because it is hard to work out what radius they should
#                    #be - the units scale differently to edge_width so need to be
#                    #trial-and-errored each time.
#
#                    # res.append(patches.Circle(poly_points[0],radius=3.2*edge_width,facecolor='k',edgecolor='k',lw=0,zorder=1))
#                    # res.append(patches.Circle(poly_points[-1],radius=3.2*edge_width,facecolor='k',edgecolor='k',lw=0,zorder=1))
#
#        return res
#
#    def plot_network_plain(self,
#                           bounds=None,
#                           ax=None,
#                           show_edges=True,
#                           show_nodes=False,
#                           edge_width=1,
#                           node_size=7,
#                           edge_col='k',
#                           node_col='r',
#                           **kwargs):
#
#        '''
#        This plots the section of the network that lies within a given bounding
#        box, inside the axes ax.
#
#        The idea is the initialise the axes elsewhere, pass them to this function,
#        and the network plot gets dumped into it - this is useful for multi-frame
#        figures, for example.
#
#        All the switches and options are fairly self-explanatory I think.
#        '''
#        ax = ax or plt.gca()
#
#        if bounds is None:
#            xmin, ymin, xmax, ymax = self.extent
#            bbox = None
#        else:
#            xmin, ymin, xmax, ymax = bounds
#
#            # create bounding box polygon for intersection tests
#            bbox = Polygon([
#                (xmin, ymin),
#                (xmax, ymin),
#                (xmax, ymax),
#                (xmin, ymax),
#                (xmin, ymin),
#            ])
#
#        if show_edges:
#            [ax.add_patch(patch) for patch in self.generate_edge_patches(bbox=bbox, edgecolor=edge_col, lw=edge_width,
#                                                                         **kwargs)]
#
#        if show_nodes:
#            node_points=[self.g.node[v]['pos'] for v in self.g]
#            if bounds:
#                node_points = [p for p in node_points if xmin <= p[0] <= xmax and ymin <= p[1] <= ymax]
#            x,y = zip(*node_points)
#            ax.scatter(x, y, c=node_col, s=node_size, zorder=5)
#
#        ax.set_xlim(xmin, xmax)
#        ax.set_ylim(ymin, ymax)
#
#    def plot_network_plain_col(self,
#                               cols,
#                               bounds=None,
#                               ax=None,
#                               show_edges=True,show_nodes=False,edge_width=1,
#                               node_size=7,edge_col='k',node_col='r'):
#
#        '''
#        This does exactly the same as above, but each edge can be given its own
#        colour. This is defined by passing a dictionary 'cols' of colours, indexed
#        by edge FID. These colours need to be worked out elsewhere (e.g. betweenness).
#
#        The way it is done is another huge fudge. A black line is produced first,
#        then a slightly thinner coloured line is placed on top of it. Horrible,
#        but I know of no other way to get the edges to have outlines, which make
#        a big difference to visual effect.
#        '''
#        ax = ax or plt.gca()
#        bounds = bounds or self.extent  # default to plotting full network
#        min_x, min_y, max_x, max_y = bounds
#
#        # create bounding box polygon for intersection tests
#        bbox = Polygon([
#            (min_x, min_y),
#            (max_x, min_y),
#            (max_x, max_y),
#            (min_x, max_y),
#            (min_x, min_y),
#        ])
#
#        if show_edges:
#            # for ls in self.lines_iter():
#            for e in self.g.edges(data=True):
#                ls = e[2]['linestring']
#                # bbox_check=[min_x<=p[0]<=max_x and min_y<=p[1]<=max_y for p in e[2]['polyline']]
#                # if any(bbox_check):
#                if ls.intersects(bbox):
#                    path=Path(ls)
#                    out_patch = patches.PathPatch(path, facecolor='none', edgecolor=edge_col, lw=edge_width)
#                    ax.add_patch(out_patch)
##                    end1=patches.Circle(poly_points[0],radius=3.2*edge_width,facecolor='k',edgecolor='k',lw=0,zorder=1)
##                    ax.add_patch(end1)
##                    end2=patches.Circle(poly_points[-1],radius=3.2*edge_width,facecolor='k',edgecolor='k',lw=0,zorder=1)
##                    ax.add_patch(end2)
#                    try:
#                        col=cols[e[2]['fid']]
#                        patch = patches.PathPatch(path, facecolor='none', edgecolor=col, lw=0.6*edge_width, zorder=2)
#                        ax.add_patch(patch)
##                        end1=patches.Circle(poly_points[0],radius=3.2*0.6*edge_width,facecolor=col,edgecolor=None,lw=0,zorder=1)
##                        ax.add_patch(end1)
##                        end2=patches.Circle(poly_points[-1],radius=3.2*0.6*edge_width,facecolor=col,edgecolor=None,lw=0,zorder=1)
##                        ax.add_patch(end2)
#                    except Exception as exc:
#                        print repr(exc)
#                        pass
#
#        if show_nodes:
#            node_points=[self.g.node[v]['pos'] for v in self.g]
#            node_points_bbox=[p for p in node_points if min_x<=p[0]<=max_x and min_y<=p[1]<=max_y]
#            x,y = zip(*node_points_bbox)
#            ax.scatter(x,y,c=node_col,s=node_size,zorder=5)
#
#        ax.set_xlim(min_x,max_x)
#        ax.set_ylim(min_y,max_y)

def read_data(filename):
    CurrentHandler=OSMHandler()
    sax.parse(filename,CurrentHandler)
    CurrentData=OSMData(CurrentHandler.nodes,CurrentHandler.ways,CurrentHandler.relations)
    return CurrentData



if __name__ == '__main__':
    import os

    this_dir = os.path.dirname(os.path.realpath(__file__))
    OSMFILE = os.path.join(this_dir, 'test_data', 'camden_fragment.osm')
#    OSMFILE = '/Users/tobydavies/osm_data_samples/camden_full.osm'
    osmdata = read_data(OSMFILE)
    o = OSMStreetNet()
    o.load_from_data(osmdata)
    
    min_x, min_y, max_x, max_y = o.extent
    
    ratio=(max_y-min_y)/(max_x-min_x)
    
    min_lon=-0.22
    max_lon=-0.10
    min_lat=51.505
    max_lat=51.58
    
    fig=plt.figure(figsize=(8,8*ratio))
    fig.subplots_adjust(left=0.01,right=0.99,bottom=0.01,top=0.99)
    ax=fig.add_subplot(111)
    
    o.plot_network(show_nodes=True)
    
    
    
    

