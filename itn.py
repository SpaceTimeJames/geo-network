# -*- coding: utf-8 -*-
"""
A number of tools for the reading and manipulation of street network data from
the ITN layer of the Ordnance Survey MasterMap product.

The first set of classes correspond exactly to the entities present in the GML
file in which the data is provided. Each is very simple and operates essentially
via tags (due to the versatility of data that can be encoded) - each entity has
an FID, some geometric information, and the rest is devolved to tags.

The first step in the workflow is to read the GML using a custom-built XML parser -
this produces these entitites, which are then held in an ITNData container class.

The main class for the street network itself (used for all processing, modelling etc)
is the ITNStreetNet - this can be built directly using an ITNData instance.

The methods of ITNStreetNet are then mainly to do with plotting, cleaning and 
routing. Most of the cleaning methods are omitted here for brevity. 

Author: Toby Davies
"""

from shapely.geometry import Point, LineString, Polygon
import xml.sax as sax
import networkx as nx
import cPickle as pk
import scipy as sp
import numpy as np
from collections import defaultdict
import bisect as bs
import pysal as psl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
from matplotlib.colors import Normalize
from matplotlib.colorbar import ColorbarBase

from distutils.version import StrictVersion


#The following classes correspond exactly to the entities present in an ITN GML
#file, and their purpose is just to store that data in a consistent way in Python.
#Each just has the FID identifier, some geometric info, and space for any
#tags in key/value format.
class Road():
    
    def __init__(self,fid,members,tags):
        self.fid = fid
        self.members = members
        self.tags = tags


class RoadNode():
    
    def __init__(self,fid,eas_nor,tags):
        self.fid = fid
        self.eas_nor = eas_nor
        self.easting, self.northing = eas_nor
        self.tags = tags


class RoadLink():
    
    def __init__(self,fid,polyline,tags):
        self.fid = fid
        self.polyline = polyline
        self.tags = tags


class RoadLinkInformation():
    
    def __init__(self,fid,roadLink_ref,tags):
        self.fid = fid
        self.roadLink_ref=roadLink_ref
        self.tags = tags
        

class ITNHandler(sax.handler.ContentHandler):
    
    '''
    This is an XML parser which is custom-built to read files provided by the ITN.
    
    It is really just the result of trial-and-error and I have NO IDEA if it is
    either 'correct' or efficient. I am a total novice in terms of XML.
    
    However, as far as I can tell, it works, and it does a better job for my purposes
    than the various tools that people use to convert ITN files to shapefiles.
    
    I'll leave it uncommented for now because: it's fairly complicated; I can't
    vouch for it being right; and it's not really important for our purposes. I
    suggest treating it as a black box unless you're particularly interested -
    will be happy to explain though.
    
    It is intended to be invoked by the read_gml() routine that appears below.
    
    The end result is nothing fancy - it just reads the XML and translates into
    the objects above. There is loads of stuff in the XML and this just scratches
    the surface - have only included stuff as necessary so far.
    
    One piece of non-trivial processing does happen. Each link is given an orientation,
    so one of the terminal nodes is described as 'negative' and the other as 'positive'.
    Useful for several reasons. For each of those nodes, it also gives a 'gradeSeparation' -
    this is an integer which encodes how high the road is at that point. The reason
    for doing this is that, for some stupid reason, every intersection between
    roads is treated as a node, even if the roads do not meet (think bridges, tunnels).
    The roads only meet if their gradeSeparations match for a given node. For reasons
    related to cleaning these features, I therefore modify the FID of each terminal
    node by appending the gradeSeparation - it becomes useful, trust me. Just so 
    you know.
    '''
    
    def __init__(self):
        sax.handler.ContentHandler.__init__(self)
        self.fid = None
        self.geometry = None
        self.tags = None
        
        self.current_type = None
        self.current_content = ''
        
        self.roads = {}
        self.roadNodes = {}
        self.roadLinks = {}
        self.roadLinkInformations = {}

    
    def startElement(self,name,attrs):
        
        if name=='osgb:Road':
            self.fid = attrs['fid']
            self.tags = {}
            self.geometry = []
            self.current_type = 'osgb:Road'
        
        elif name=='osgb:networkMember' and self.current_type=='osgb:Road':
            self.geometry.append(attrs['xlink:href'][1:])
        
        elif name=='osgb:RoadNode':
            self.fid = attrs['fid']
            self.tags = {}
            self.current_type = 'osgb:RoadNode'
        
        elif name=='osgb:RoadLink':
            self.fid = attrs['fid']
            self.tags = {}
            self.geometry = []
            self.current_type = 'osgb:RoadLink'
        
        elif name=='osgb:directedNode' and self.current_type=='osgb:RoadLink':
            if attrs['orientation']=='-':
                
                if 'gradeSeparation' in attrs:
                    self.tags['gradeSeparation_neg']=attrs['gradeSeparation']
                else:
                    self.tags['gradeSeparation_neg']=str(0)
                
                self.tags['orientation_neg']=attrs['xlink:href'][1:]+'_'+self.tags['gradeSeparation_neg']
                    
            elif attrs['orientation']=='+':
                
                if 'gradeSeparation' in attrs:
                    self.tags['gradeSeparation_pos']=attrs['gradeSeparation']
                else:
                    self.tags['gradeSeparation_pos']=str(0)
                
                self.tags['orientation_pos']=attrs['xlink:href'][1:]+'_'+self.tags['gradeSeparation_pos']
            
        elif name=='osgb:RoadLinkInformation':
            self.fid = attrs['fid']
            self.tags = {}
            self.current_type = 'osgb:RoadLinkInformation'
        
        elif name=='osgb:referenceToRoadLink' and self.current_type=='osgb:RoadLinkInformation':
            self.geometry=attrs['xlink:href'][1:]
        
        self.current_content=''
    
    
    def characters(self,content):
        self.current_content = self.current_content+content
        
    
    def endElement(self,name):
        if name=='osgb:Road':
            self.roads[self.fid] = Road(self.fid,self.geometry,self.tags)
            self.reset()
        
        elif name=='osgb:roadName' and self.current_type=='osgb:Road':
            self.tags['roadName']=self.current_content
        
        #The descriptiveGroup is 'Named Road', 'A Road', 'Motorway' etc
        elif name=='osgb:descriptiveGroup' and self.current_type=='osgb:Road':
            self.tags['descriptiveGroup']=self.current_content
        
        #The descriptiveTerm is 'Primary Route' etc, otherwise NULL
        elif name=='osgb:descriptiveTerm' and self.current_type=='osgb:Road':
            self.tags['descriptiveTerm']=self.current_content
        
        elif name=='osgb:RoadNode':
            self.roadNodes[self.fid] = RoadNode(self.fid,self.geometry,self.tags)
            self.reset()
        
        elif name=='gml:coordinates' and self.current_type=='osgb:RoadNode':
            coords=self.current_content.split(',')
            self.geometry=tuple(map(float,coords))
        
        elif name=='osgb:RoadLink':
            self.roadLinks[self.fid] = RoadLink(self.fid,self.geometry,self.tags)
            self.reset()
        
        elif name=='osgb:descriptiveTerm' and self.current_type=='osgb:RoadLink':
            self.tags['descriptiveTerm']=self.current_content
        
        elif name=='osgb:natureOfRoad' and self.current_type=='osgb:RoadLink':
            self.tags['natureOfRoad']=self.current_content
        
        elif name=='osgb:length' and self.current_type=='osgb:RoadLink':
            self.tags['length']=float(self.current_content)
        
        elif name=='gml:coordinates' and self.current_type=='osgb:RoadLink':
            points=self.current_content.split()
            points_coords=[p.split(',') for p in points]
            self.geometry=[tuple(map(float,p)) for p in points_coords]
        
        elif name=='osgb:RoadLinkInformation':
            self.roadLinkInformations[self.fid] = RoadLinkInformation(self.fid,self.geometry,self.tags)
            self.reset()
        
        elif name=='osgb:classification' and self.current_type=='osgb:RoadLinkInformation':
            self.tags['classification']=self.current_content
        
        elif name=='osgb:distanceFromStart' and self.current_type=='osgb:RoadLinkInformation':
            self.tags['distanceFromStart']=self.current_content
        
        elif name=='osgb:feet' and self.current_type=='osgb:RoadLinkInformation':
            self.tags['feet']=self.current_content
        
        elif name=='osgb:inches' and self.current_type=='osgb:RoadLinkInformation':
            self.tags['inches']=self.current_content
        
    
    def reset (self):
        self.fid = None
        self.geometry = None
        self.tags = None
        
        self.current_type = None




class ITNData():
    
    '''
    This is just a container class for the output of a GML parse.
    
    The reason for it existing is so that it can be saved directly to avoid having
    to do the parsing (which gets slow for big files) every time.
    '''
    
    def __init__(self,roads,roadNodes,roadLinks,roadLinkInformations):
        self.roads = roads
        self.roadNodes = roadNodes
        self.roadLinks = roadLinks
        self.roadLinkInformations = roadLinkInformations
        
   
    def save(self,filename):
        f=open(filename,'wb')
        pk.dump(self,f)
        f.close()


class ITNStreetNet(object):
    
    '''
    This is the main street network object derived from the ITN.
    
    It is always initialised empty, and can then be provided data in two ways: by
    building a network from an ITNData instance, or by inheriting an already-built
    network from elsewhere.
    
    The reason for the latter is cleaning - sometimes it is useful to produce a 
    'new' ITNStreetNet derived from a previous one.
    
    The graph used is a multigraph - sometimes multiple edges exist between nodes
    (a lay-by is the most obvious example). That causes a fair bit of pain, but
    there is no alternative.
    
    The graph is undirected - defer dealing with routing information for now, although
    often undirected is preferable anyway.
    
    NetworkX networks are stored using a dictionary-of-dictionaries model. Each edge
    is technically a triple - the two terminal nodes and a set of attributes. There
    are loads of ways of accessing and manipulating these and iterating over them.
    
    All cleaning routines stripped out for now because they're uncommented, ugly
    and bloated.
    '''
    
    def __init__(self):
        '''
        This just initialises a fresh empty network in each new class. Will be
        overwritten but just ensures that stuff won't break.
        '''
        
        self.g=nx.MultiGraph()
    
    
    def load_from_data(self,Data):
        '''
        ITNData passed to this and the network is built using routines below.
        Positional info is also taken; can be useful for plotting.
        '''
        
        print 'Building the network'
        self.build_network(Data)
        
        print 'Building position dictionary'
        self.build_posdict(Data)
        
    
    def inherit(self,g):
        
        self.g=g
    
    
    def build_network(self,Data):
        
        g=nx.MultiGraph()
        
        for roadLink_fid, roadLink_inst in Data.roadLinks.iteritems():
            
            #In network terms, the roadlink is just encoded as a simple edge between
            #the two terminal nodes, but the full polyline geometry is included
            #as an attribute of the link, as is the FID.
            
            atts = roadLink_inst.tags
            # replacing this with a pre-built Shapely Linestring
            # atts['polyline'] = roadLink_inst.polyline
            atts['fid'] = roadLink_fid
            atts['linestring'] = LineString(roadLink_inst.polyline)


            #This if statement just checks that both terminal nodes are in the roadNodes
            #dataset. Sometimes, around the edges, they are not - this causes problems
            #down the line so such links are omitted. The [:-2] is to ignore the
            #gradeSeparation tag that I added when parsing.
            
            if atts['orientation_neg'][:-2] in Data.roadNodes and atts['orientation_pos'][:-2] in Data.roadNodes:
                g.add_edge(atts['orientation_neg'],atts['orientation_pos'],key=roadLink_fid,attr_dict=atts)
        
        #Only want the largest connected component - sometimes fragments appear
        #round the edge - so take that.
        # NB this interface has changed with NetworkX v1.9
        if StrictVersion(nx.__version__) >= StrictVersion('1.9'):
            g = sorted(nx.connected_component_subgraphs(g), key=len, reverse=True)[0]
        else:
            g=nx.connected_component_subgraphs(g)[0]

        self.g=g
    
    
    def build_posdict(self,Data):
        '''
        Each node gets an attribute added for its geometric position. This is only
        really useful for plotting.
        '''
        
        for v in self.g:
            self.g.node[v]['pos']=Data.roadNodes[v[:-2]].eas_nor


    def edges(self, bpoly=None):
        '''
        Get all edges in the network.  Optionally return only those that intersect the provided bounding polygon
        '''
        if bpoly:
            return [x for x in self.g.edges(data=True) if bpoly.intersects(x[2]['linestring'])]
        else:
            return self.g.edges(data=True)

    
    def plot_network_plain(self,min_x,max_x,min_y,max_y, ax=None,
                           show_edges=True,show_nodes=False,edge_width=1,
                           node_size=7,edge_col='k',node_col='r'):
        
        '''
        This plots the section of the network that lies within a given bounding
        box, inside the axes ax.
        
        The idea is the initialise the axes elsewhere, pass them to this function,
        and the network plot gets dumped into it - this is useful for multi-frame
        figures, for example.
        
        All the switches and options are fairly self-explanatory I think.
        '''
        ax = ax or plt.gca()
        
        if show_edges:
            for e in self.g.edges(data=True):
                
                bbox_check=[min_x<=p[0]<=max_x and min_y<=p[1]<=max_y for p in e[2]['polyline']]
                
                #This checks that at least some of the line lies within the bounding
                #box. This is to avoid creating unnecessary lines which will not
                #actually be seen.
                
                if any(bbox_check):
                    path=Path(e[2]['polyline'])
                    patch = patches.PathPatch(path, facecolor='none', edgecolor=edge_col, lw=edge_width)
                    ax.add_patch(patch)
                    
                    #These circles are a massive fudge to give the lines 'rounded'
                    #ends. They look nice, but best to only do them at the last
                    #minute because it is hard to work out what radius they should
                    #be - the units scale differently to edge_width so need to be
                    #trial-and-errored each time.
                    
#                    end1=patches.Circle(poly_points[0],radius=3.2*edge_width,facecolor='k',edgecolor='k',lw=0,zorder=1)
#                    ax.add_patch(end1)
#                    end2=patches.Circle(poly_points[-1],radius=3.2*edge_width,facecolor='k',edgecolor='k',lw=0,zorder=1)
#                    ax.add_patch(end2)
        
        if show_nodes:
            node_points=[self.g.node[v]['pos'] for v in self.g]
            node_points_bbox=[p for p in node_points if min_x<=p[0]<=max_x and min_y<=p[1]<=max_y]
            x,y = zip(*node_points_bbox)
            ax.scatter(x,y,c=node_col,s=node_size,zorder=5)
        
        ax.set_xlim(min_x,max_x)
        ax.set_ylim(min_y,max_y)
    
    
    def plot_network_plain_col(self,min_x,max_x,min_y,max_y,cols, ax=None,
                           show_edges=True,show_nodes=False,edge_width=1,
                           node_size=7,edge_col='k',node_col='r'):
        
        '''
        This does exactly the same as above, but each edge can be given its own 
        colour. This is defined by passing a dictionary 'cols' of colours, indexed
        by edge FID. These colours need to be worked out elsewhere (e.g. betweenness).
        
        The way it is done is another huge fudge. A black line is produced first,
        then a slightly thinner coloured line is placed on top of it. Horrible,
        but I know of no other way to get the edges to have outlines, which make
        a big difference to visual effect.
        '''
        ax = ax or plt.gca()
        
        if show_edges:
            for e in self.g.edges(data=True):
                
                bbox_check=[min_x<=p[0]<=max_x and min_y<=p[1]<=max_y for p in e[2]['polyline']]
                
                if any(bbox_check):
                    path=Path(e[2]['polyline'])
                    out_patch = patches.PathPatch(path, facecolor='none', edgecolor=edge_col, lw=edge_width)
                    ax.add_patch(out_patch)
#                    end1=patches.Circle(poly_points[0],radius=3.2*edge_width,facecolor='k',edgecolor='k',lw=0,zorder=1)
#                    ax.add_patch(end1)
#                    end2=patches.Circle(poly_points[-1],radius=3.2*edge_width,facecolor='k',edgecolor='k',lw=0,zorder=1)
#                    ax.add_patch(end2)
                    try:
                        col=cols[e[2]['fid']]
                        patch = patches.PathPatch(path, facecolor='none', edgecolor=col, lw=0.6*edge_width, zorder=2)
                        ax.add_patch(patch)
#                        end1=patches.Circle(poly_points[0],radius=3.2*0.6*edge_width,facecolor=col,edgecolor=None,lw=0,zorder=1)
#                        ax.add_patch(end1)
#                        end2=patches.Circle(poly_points[-1],radius=3.2*0.6*edge_width,facecolor=col,edgecolor=None,lw=0,zorder=1)
#                        ax.add_patch(end2)
                    except:
                        pass
        
        if show_nodes:
            node_points=[self.g.node[v]['pos'] for v in self.g]
            node_points_bbox=[p for p in node_points if min_x<=p[0]<=max_x and min_y<=p[1]<=max_y]
            x,y = zip(*node_points_bbox)
            ax.scatter(x,y,c=node_col,s=node_size,zorder=5)
        
        ax.set_xlim(min_x,max_x)
        ax.set_ylim(min_y,max_y)
    
    
    def within_boundary(self,poly,outer_buffer=0):
        
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
        for e in self.g.edges(data=True):
            #Make a shapely polyline for each
            edge_line=LineString(e[2]['polyline'])
            
            #Check intersection
            if edge_line.intersects(boundary):
                #Add edge to new graph
                g_new.add_edge(e[0],e[1],key=e[2]['fid'],attr_dict=e[2])
        
        #Add all nodes to the new posdict
        for v in g_new:
            g_new.node[v]['pos']=self.g.node[v]['pos']
        
        #Make the new class and inherit the network        
        net_new=ITNStreetNet()
        net_new.inherit(g_new)
        
        return net_new
    
    
    def save(self,filename):
        '''
        Just saves the network for convenience.
        '''
        f=open(filename,'wb')
        pk.dump(self,f)
        f.close()
    
    
    def bin_edges(self,min_x,max_x,min_y,max_y,gridsize):
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
        
        #Set up grids and arrays
        x_grid=sp.arange(min_x,max_x,gridsize)
        y_grid=sp.arange(min_y,max_y,gridsize)
        
        #Initialise the lookup
        edge_locator=defaultdict(list)
        
        #Loop segments
        for e in self.g.edges_iter(data=True):
            #Produce shapely polyline
            # edge_line=LineString(e[2]['polyline'])
            edge_line = e[2]['linestring']
            
            #Get bounding box of polyline
            (bbox_min_x, bbox_min_y, bbox_max_x, bbox_max_y)=edge_line.bounds
            
            #Bin bbox extremities
            bbox_min_x_loc=bs.bisect_left(x_grid,bbox_min_x)
            bbox_max_x_loc=bs.bisect_left(x_grid,bbox_max_x)
            bbox_min_y_loc=bs.bisect_left(y_grid,bbox_min_y)
            bbox_max_y_loc=bs.bisect_left(y_grid,bbox_max_y)
            
            #Go through every cell covered by this bbox, augmenting lookup
            for i in range(bbox_min_x_loc,bbox_max_x_loc+1):
                for j in range(bbox_min_y_loc,bbox_max_y_loc+1):
                    edge_locator[(i,j)].append(e)
            
        return x_grid,y_grid,edge_locator
    
    
    def closest_segments_euclidean(self,x,y,x_grid,y_grid,edge_locator,radius=50):
        '''
        Snap a point to the closest segment.
        
        Needs to be passed the output from a previous run of bin_edges().
        
        The radius argument allows an upper limit on the snap distance to be imposed -
        'if there are no streets within radius metres, return Null. This is useful
        for some purposes.
        '''
        
        #Produce a shapely point and find which cell it lies in.
        point=Point(x,y)
        
        x_loc=bs.bisect_left(x_grid,x)
        y_loc=bs.bisect_left(y_grid,y)
        
        #Go round this cell and all neighbours (9 in total) collecting all edges
        #which could intersect with any of these.
        near_edges=[]
        
        for i in [-1,0,1]:
            for j in [-1,0,1]:
                for e in edge_locator[(x_loc+i,y_loc+j)]:
                    if e not in near_edges:
                        near_edges.append(e)
        
        #near_edges now contains all candidates for closest edge
        
        #Calculate the distances to each
        # edge_distances=[point.distance(LineString(e[2]['polyline'])) for e in near_edges]
        edge_distances=[point.distance(e[2]['linestring']) for e in near_edges]
        
        #Order the edges according to proximity, omitting those which are further than radius away
        edge_hierarchy=[x for x in sorted(zip(near_edges,edge_distances),key=lambda x: x[1]) if x[1]<radius]
        
        if len(edge_hierarchy)==0:
            #This may be an empty list, in which case bail out            
            return None
        
        else:
            #Otherwise, do various proximity calculations
            
            #Already know the closest edge, and how far it is from the point
            closest_edge,snap_dist=edge_hierarchy[0]
            
            # closest_polyline=LineString(closest_edge[2]['polyline'])
            closest_polyline=closest_edge[2]['linestring']
            
            #dist_along is a lookup, indexed by each of the terminal nodes of closest_edge,
            #which gives the distance from that node to the point on the line to which
            #the original point snaps to. If that makes sense.
            #The polyline is specified from negative to positive orientation BTW.
            dist_along={}
            
            dist_along[closest_edge[2]['orientation_neg']]=closest_polyline.project(point)
            dist_along[closest_edge[2]['orientation_pos']]=closest_polyline.length-closest_polyline.project(point)
                
            return closest_edge,dist_along,snap_dist


    def dist_between_points(self, dist_along1, dist_along2):
        '''
        Gives the minimum network distance between two points, given the dist_along
        output for each of them from closest_segments_euclidean().
        
        Obviously the shortest path from p1 to p2 takes the form of:
            - travel from p1 to one of its terminal nodes
            - travel from this terminal node to one of the terminal nodes of p2
            - travel from this terminal node to p2
        
        There are clearly 4 ways this can happen (2 options for terminal node in
        each case); simply enumerate these.
        '''
        
        distances=[]
        
        for v in dist_along1:
            for w in dist_along2:
                #Get the network distance between terminal nodes
                network_distance=nx.dijkstra_path_length(self.g,v,w,'length')
                
                #Add the extra distance at each end of the route
                total_distance=network_distance+dist_along1[v]+dist_along2[w]
                
                #Add to the list
                distances.append(total_distance)
        
        return min(distances)

    ### ADDED BY GABS
    def lines_iter(self):
        """
        Returns a generator that iterates over all edge linestrings.
        This is useful for various spatial operations.
        """
        for e in self.g.edges_iter(data=True):
            yield e[2]['linestring']

    def closest_segments_euclidean_brute_force(self, x, y, radius=None):
        pt = Point(x, y)
        if radius:
            bpoly = pt.buffer(radius)
        else:
            bpoly = None

        edges = self.edges(bpoly=bpoly)
        if not len(edges):
            # no valid edges found, bail.
            return None

        snap_distances = [x.distance(pt) for x in self.lines_iter()]
        snap_distance = min(snap_distances)
        closest_edge = edges[snap_distances.index(snap_distance)]

        da = closest_edge[2]['linestring'].project(pt)
        dist_along={
            closest_edge[2]['orientation_neg']: da,
            closest_edge[2]['orientation_pos']: closest_edge[2]['linestring'].length - da,
        }

        return closest_edge, dist_along, snap_distance

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


def read_gml(filename):
    CurrentHandler=ITNHandler()
    sax.parse(filename,CurrentHandler)
    CurrentData=ITNData(CurrentHandler.roads,CurrentHandler.roadNodes,
                        CurrentHandler.roadLinks,CurrentHandler.roadLinkInformations)
    return CurrentData


if __name__ == '__main__':

    from settings import DATA_DIR
    import os

    ITNFILE = os.path.join(DATA_DIR, 'network_data/itn_sample', 'mastermap-itn_417209_0_brixton_sample.gml')

    # A little demo

    #Just build the network as usual
    itndata = read_gml(ITNFILE)
    g = ITNStreetNet()
    g.load_from_data(itndata)

    # generate some random points inside camden
    import numpy as np
    xmin, ymin, xmax, ymax = g.extent
    xs = np.random.rand(100)*(xmax - xmin) + xmin
    ys = np.random.rand(100)*(ymax - ymin) + ymin
    net_pts = [g.closest_segments_euclidean_brute_force(x, y)[1] for (x, y) in zip(xs, ys)]