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
from collections import defaultdict
import bisect as bs
import datetime as dt
#import pysal as psl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
#from matplotlib.colors import Normalize
#from matplotlib.colorbar import ColorbarBase


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


class RoadRouteInformation():
    
    def __init__(self,fid,route_members,tags):
        self.fid = fid
        self.route_members=route_members
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
        self.roadRouteInformations = {}

    
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
        
        elif name=='osgb:RoadRouteInformation':
            self.fid = attrs['fid']
            self.tags = {}
            self.geometry = {}
            self.current_type = 'osgb:RoadRouteInformation'
        
        elif name=='osgb:directedLink' and self.current_type=='osgb:RoadRouteInformation':
            #For 'One way', the traffic flows TOWARDS the given orientation
            self.geometry[attrs['orientation']]=attrs['xlink:href'][1:]
        
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
        
        elif name=='osgb:RoadRouteInformation':
            self.roadRouteInformations[self.fid] = RoadRouteInformation(self.fid,self.geometry,self.tags)
            self.reset()
        
        elif name=='osgb:instruction' and self.current_type=='osgb:RoadRouteInformation':
            self.tags['instruction']=self.current_content
        
        elif name=='osgb:classification' and self.current_type=='osgb:RoadRouteInformation':
            self.tags['classification']=self.current_content
        
        elif name=='osgb:distanceFromStart' and self.current_type=='osgb:RoadRouteInformation':
            self.tags['distanceFromStart']=self.current_content
        
        elif name=='osgb:namedTime' and self.current_type=='osgb:RoadRouteInformation':
            self.tags['namedTime']=self.current_content
        
        elif name=='osgb:type' and self.current_type=='osgb:RoadRouteInformation':
            self.tags['type']=self.current_content
        
        elif name=='osgb:use' and self.current_type=='osgb:RoadRouteInformation':
            self.tags['use']=self.current_content
        
        elif name=='osgb:startTime' and self.current_type=='osgb:RoadRouteInformation':
            datetime_object=dt.datetime.strptime(self.current_content, '%H:%M:%S')
            self.tags['startTime']=datetime_object.time()
        
        elif name=='osgb:endTime' and self.current_type=='osgb:RoadRouteInformation':
            datetime_object=dt.datetime.strptime(self.current_content, '%H:%M:%S')
            self.tags['endTime']=datetime_object.time()
    
    
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
    
    def __init__(self,roads,roadNodes,roadLinks,roadLinkInformations,roadRouteInformations):
        self.roads = roads
        self.roadNodes = roadNodes
        self.roadLinks = roadLinks
        self.roadLinkInformations = roadLinkInformations
        self.roadRouteInformations = roadRouteInformations
        
   
    def save(self,filename):
        f=open(filename,'wb')
        pk.dump(self,f)
        f.close()


class ITNStreetNet():
    
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
        
        print 'Building routing network'
        self.build_routing_network()
        
    
    def inherit(self,g):
        
        self.g=g
        self.build_routing_network()
    
    
    def build_network(self,Data):
        
        g=nx.MultiGraph()
        
        for roadLink_fid in Data.roadLinks:
            
            #In network terms, the roadlink is just encoded as a simple edge between
            #the two terminal nodes, but the full polyline geometry is included
            #as an attribute of the link, as is the FID.
            
            atts=Data.roadLinks[roadLink_fid].tags
            atts['polyline']=Data.roadLinks[roadLink_fid].polyline
            atts['fid']=roadLink_fid
            
            #This if statement just checks that both terminal nodes are in the roadNodes
            #dataset. Sometimes, around the edges, they are not - this causes problems
            #down the line so such links are omitted. The [:-2] is to ignore the
            #gradeSeparation tag that I added when parsing.
            
            if atts['orientation_neg'][:-2] in Data.roadNodes and atts['orientation_pos'][:-2] in Data.roadNodes:
                
                g.add_edge(atts['orientation_neg'],atts['orientation_pos'],key=atts['fid'],attr_dict=atts)
        
        #Only want the largest connected component - sometimes fragments appear
        #round the edge - so take that. 
        g=list(nx.connected_component_subgraphs(g))[0]
        
        #Now record one-way status for every segment
        #The idea is to go through all roadRouteInformations looking for the ones
        #that correspond to a one-way directive
        for roadRouteInformation_fid in Data.roadRouteInformations:
            
            #Get the actual content of the roadRouteInformation
            atts=Data.roadRouteInformations[roadRouteInformation_fid].tags
            
            #A one-way item has an 'instruction' attribute with value 'One Way',
            #so look for that
            if 'instruction' in atts:
            
                if atts['instruction']=='One Way':
                    
                    #The orientation is either positive of negative, and in either
                    #case the roadLink_fid to which it refers is specified
                    if '+' in Data.roadRouteInformations[roadRouteInformation_fid].route_members:
                        
                        roadLink_fid=Data.roadRouteInformations[roadRouteInformation_fid].route_members['+']
                        
                        orientation='pos'
                        
                    else:
                        
                        roadLink_fid=Data.roadRouteInformations[roadRouteInformation_fid].route_members['-']
                        
                        orientation='neg'
                    
                    #Get the relevant terminal nodes so that we can look up the edge
                    v0=Data.roadLinks[roadLink_fid].tags['orientation_pos']
                    v1=Data.roadLinks[roadLink_fid].tags['orientation_neg']
                    
                    #A try method is used here because sometimes the link to which
                    #the instruction refers is not actually in the graph (usually
                    #an edge effect) - the lookup would throw an error in such cases.
                    try:
                        g.edge[v0][v1][roadLink_fid]['one_way']=orientation
                    except:
                        pass
            
        self.g=g
    
    
    def build_posdict(self,Data):
        '''
        Each node gets an attribute added for its geometric position. This is only
        really useful for plotting.
        '''
        
        for v in self.g:
            self.g.node[v]['loc']=Data.roadNodes[v[:-2]].eas_nor
        
    
    def build_routing_network(self):
        '''
        This builds a second graph for the class representing the routing information.
        
        It is a directed multigraph in which direction represents allowed travel.
        Edges are present in both directions if travel is two-way, otherwise only
        one is included.
        
        All other attributes are exactly the same as the underlying g, and inherited
        as such.
        '''
        
        g_routing=nx.MultiDiGraph()
        
        #Loop the edges of g and assess the one-way status of each
        for n1,n2,fid,attr in self.g.edges(data=True,keys=True):
            
            #If one_way attribute is present, only add an edge in the correct direction
            if 'one_way' in attr:
                
                if attr['one_way']=='pos':
                    g_routing.add_edge(attr['orientation_neg'],attr['orientation_pos'],key=fid,attr_dict=attr)
                
                else:
                    g_routing.add_edge(attr['orientation_pos'],attr['orientation_neg'],key=fid,attr_dict=attr)
            
            #If the attribute is absent, add edges in both directions
            else:
                g_routing.add_edge(attr['orientation_neg'],attr['orientation_pos'],key=fid,attr_dict=attr)
                g_routing.add_edge(attr['orientation_pos'],attr['orientation_neg'],key=fid,attr_dict=attr)
                
        for v in g_routing:
            g_routing.node[v]['loc']=self.g.node[v]['loc']
        
        self.g_routing=g_routing
            
    
    def plot_network_plain(self,ax,min_x,max_x,min_y,max_y,
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
        
        if show_edges:
            for n1,n2,fid,attr in self.g.edges(data=True,keys=True):
                
                bbox_check=[min_x<=p[0]<=max_x and min_y<=p[1]<=max_y for p in attr['polyline']]
                
                #This checks that at least some of the line lies within the bounding
                #box. This is to avoid creating unnecessary lines which will not
                #actually be seen.
                
                if any(bbox_check):
                    path=Path(attr['polyline'])
                    patch = patches.PathPatch(path, facecolor='none', edgecolor=edge_col, lw=edge_width)
                    ax.add_patch(patch)
                    
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
        
        if show_nodes:
            node_points=[self.g.node[v]['loc'] for v in self.g]
            node_points_bbox=[p for p in node_points if min_x<=p[0]<=max_x and min_y<=p[1]<=max_y]
            x,y = zip(*node_points_bbox)
            ax.scatter(x,y,c=node_col,s=node_size,zorder=5)
        
        ax.set_xlim(min_x,max_x)
        ax.set_ylim(min_y,max_y)
    
    
    def plot_network_plain_col(self,ax,min_x,max_x,min_y,max_y,cols,
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
        
        if show_edges:
            for n1,n2,fid,attr in self.g.edges(data=True,keys=True):
                
                bbox_check=[min_x<=p[0]<=max_x and min_y<=p[1]<=max_y for p in attr['polyline']]
                
                if any(bbox_check):
                    path=Path(attr['polyline'])
                    out_patch = patches.PathPatch(path, facecolor='none', edgecolor=edge_col, lw=edge_width)
                    ax.add_patch(out_patch)
                    
#                    end1=patches.Circle(attr['polyline'][0],radius=0.4*edge_width,facecolor='k',edgecolor='k',lw=0,zorder=1)
#                    ax.add_patch(end1)
#                    end2=patches.Circle(attr['polyline'][-1],radius=0.4*edge_width,facecolor='k',edgecolor='k',lw=0,zorder=1)
#                    ax.add_patch(end2)
                    
                    try:
                        col=cols[fid]
                        patch = patches.PathPatch(path, facecolor='none', edgecolor=col, lw=0.6*edge_width, zorder=2)
                        ax.add_patch(patch)
                        
#                        end1=patches.Circle(attr['polyline'][0],radius=0.4*0.6*edge_width,facecolor=col,edgecolor=None,lw=0,zorder=1)
#                        ax.add_patch(end1)
#                        end2=patches.Circle(attr['polyline'][-1],radius=0.4*0.6*edge_width,facecolor=col,edgecolor=None,lw=0,zorder=1)
#                        ax.add_patch(end2)
                    except:
                        pass
        
        if show_nodes:
            node_points=[self.g.node[v]['loc'] for v in self.g]
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
        for n1,n2,fid,attr in self.g.edges(data=True,keys=True):
            #Make a shapely polyline for each
            edge_line=LineString(attr['polyline'])
            
            #Check intersection
            if edge_line.intersects(boundary):
                #Add edge to new graph
                g_new.add_edge(n1,n2,key=fid,attr_dict=attr)
        
        #Add all nodes to the new posdict
        for v in g_new:
            g_new.node[v]['loc']=self.g.node[v]['loc']
        
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
    
    
    def build_grid_edge_index(self,min_x,max_x,min_y,max_y,gridsize):
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
        edge_index=defaultdict(list)
        
        #Loop edges
        for n1,n2,fid,attr in self.g.edges(data=True,keys=True):
            #Produce shapely polyline
            edge_line=LineString(attr['polyline'])
            
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
        point=Point(x,y)
        
        x_loc=bs.bisect_left(grid_edge_index.x_grid,x)
        y_loc=bs.bisect_left(grid_edge_index.y_grid,y)
        
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
        candidate_edge_distances=[point.distance(LineString(self.g[n1][n2][fid]['polyline'])) for (n1,n2,fid) in candidate_edges]
                
        #Order the edges according to proximity, omitting those which are further than radius away
        valid_edges_distances=[w for w in sorted(zip(candidate_edges,candidate_edge_distances),key=lambda w: w[1]) if w[1]<radius]
        
        closest_edges=[]
        
        for (n1,n2,fid),snap_distance in valid_edges_distances[:max_edges]:
            
            #Do various proximity calculations
            
            polyline=LineString(self.g[n1][n2][fid]['polyline'])
            
            #node_dist is a lookup, indexed by each of the terminal nodes of closest_edge,
            #which gives the distance from that node to the point on the line to which
            #the original point snaps to. If that makes sense.
            #The polyline is specified from negative to positive orientation BTW.
            node_dist={}
            
            node_dist[self.g[n1][n2][fid]['orientation_neg']]=polyline.project(point)
            node_dist[self.g[n1][n2][fid]['orientation_pos']]=polyline.length-polyline.project(point)
            
            net_point=NetPoint(self,(n1,n2,fid),node_dist)
            
            closest_edges.append((net_point,snap_distance))
        
        return closest_edges
                
        
    def path_undirected(self,net_point1,net_point2):
        
        n1_1,n2_1,fid_1=net_point1.edge
        n1_2,n2_2,fid_2=net_point2.edge
        
        node_dist1=net_point1.node_dist
        node_dist2=net_point2.node_dist
        
        if fid_1==fid_2:
            
            dist_diff=node_dist2[n1_1]-node_dist1[n1_1]
            
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
    
    
    def path_directed(self,net_point1,net_point2):
        
        n1_1,n2_1,fid_1=net_point1.edge
        n1_2,n2_2,fid_2=net_point2.edge
        
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
    


class NetPoint():
    
    def __init__(self,street_net,edge,node_dist):
        
        self.street_net = street_net
        self.edge = edge
        self.node_dist = node_dist



class NetPath():
    
    def __init__(self,start,end,edges,distances,nodes):
        
        self.start = start
        self.end = end
        self.edges = edges
        self.distances = distances
        self.nodes = nodes
        
        self.length = sum(distances)
        
        if len(distances)!=len(edges):
            print 'Path mismatch: distance list wrong length'
        
        if len(nodes)!=len(edges)-1:
            print 'Path mismatch: node list wrong length'


class GridEdgeIndex():
    
    def __init__(self,x_grid,y_grid,edge_index):
        
        self.x_grid=x_grid
        self.y_grid=y_grid
        self.edge_index=edge_index
        


def read_gml(filename):
    CurrentHandler=ITNHandler()
    sax.parse(filename,CurrentHandler)
    CurrentData=ITNData(CurrentHandler.roads,CurrentHandler.roadNodes,
                        CurrentHandler.roadLinks,CurrentHandler.roadLinkInformations,
                        CurrentHandler.roadRouteInformations)
    return CurrentData



'''Just a daft testing section that I use for sanity purposes'''

#TestData=read_gml('../network_data/mastermap-itn_417209_0_brixton_sample.gml')
##TestData=read_gml('../network_data/mastermap-itn_544003_0_camden_buff2000.gml')
#
#
#CurrentNet=ITNStreetNet()
#CurrentNet.load_from_data(TestData)
#
#grid_edge_index=CurrentNet.build_grid_edge_index(530850,531900,174550,175500,50)
#
#
##Four test points - 1 and 3 on same segment, 2 on neighbouring segment, 4 long way away.
##5 and 6 are created so that there are 2 paths of almost-equal length between them - they
##lie on opposite sides of a 'square'
#net_point1,snap_dist=CurrentNet.closest_edges_euclidean(531190,175214,grid_edge_index)[0]
#net_point2,snap_dist=CurrentNet.closest_edges_euclidean(531149,175185,grid_edge_index)[0]
#net_point3,snap_dist=CurrentNet.closest_edges_euclidean(531210,175214,grid_edge_index)[0]
#net_point4,snap_dist=CurrentNet.closest_edges_euclidean(531198,174962,grid_edge_index)[0]
#net_point5,snap_dist=CurrentNet.closest_edges_euclidean(531090,175180,grid_edge_index)[0]
#net_point6,snap_dist=CurrentNet.closest_edges_euclidean(531110,175110,grid_edge_index)[0]
#net_point7,snap_dist=CurrentNet.closest_edges_euclidean(531050,175300,grid_edge_index)[0]
#net_point8,snap_dist=CurrentNet.closest_edges_euclidean(530973,175210,grid_edge_index)[0]
#net_point9,snap_dist=CurrentNet.closest_edges_euclidean(530975,175217,grid_edge_index)[0]
#
#
#print CurrentNet.path_undirected(net_point1,net_point2).length
#print CurrentNet.path_undirected(net_point2,net_point1).length
#
#print CurrentNet.path_directed(net_point1,net_point2).length
#print CurrentNet.path_directed(net_point2,net_point1).length


