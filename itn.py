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
from network.utils import network_point_coverage

from shapely.geometry import Point, LineString, Polygon
import xml.sax as sax
import datetime
from matplotlib import pyplot as plt
import networkx as nx
import cPickle

from streetnet import StreetNet
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
            datetime_object=datetime.datetime.strptime(self.current_content, '%H:%M:%S')
            self.tags['startTime']=datetime_object.time()

        elif name=='osgb:endTime' and self.current_type=='osgb:RoadRouteInformation':
            datetime_object=datetime.datetime.strptime(self.current_content, '%H:%M:%S')
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
        with open(filename, 'wb') as f:
            cPickle.dump(self, f)



class ITNStreetNet(StreetNet):
    
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

    def build_network(self, data):

        g=nx.MultiGraph()

        for roadLink_fid in data.roadLinks:

            #In network terms, the roadlink is just encoded as a simple edge between
            #the two terminal nodes, but the full polyline geometry is included
            #as an attribute of the link, as is the FID.

            atts = data.roadLinks[roadLink_fid].tags
            # atts['polyline'] = data.roadLinks[roadLink_fid].polyline
            atts['linestring'] = LineString(data.roadLinks[roadLink_fid].polyline)
            atts['length'] = atts['linestring'].length
            atts['fid']=roadLink_fid

            #This if statement just checks that both terminal nodes are in the roadNodes
            #dataset. Sometimes, around the edges, they are not - this causes problems
            #down the line so such links are omitted. The [:-2] is to ignore the
            #gradeSeparation tag that I added when parsing.

            if atts['orientation_neg'][:-2] in data.roadNodes and atts['orientation_pos'][:-2] in data.roadNodes:

                g.add_edge(atts['orientation_neg'],atts['orientation_pos'],key=atts['fid'],attr_dict=atts)

        #Only want the largest connected component - sometimes fragments appear
        #round the edge - so take that.
        g=list(nx.connected_component_subgraphs(g))[0]

        #Now record one-way status for every segment
        #The idea is to go through all roadRouteInformations looking for the ones
        #that correspond to a one-way directive
        for roadRouteInformation_fid in data.roadRouteInformations:

            #Get the actual content of the roadRouteInformation
            atts = data.roadRouteInformations[roadRouteInformation_fid].tags

            #A one-way item has an 'instruction' attribute with value 'One Way',
            #so look for that
            if 'instruction' in atts:

                if atts['instruction']=='One Way':

                    #The orientation is either positive of negative, and in either
                    #case the roadLink_fid to which it refers is specified
                    if '+' in data.roadRouteInformations[roadRouteInformation_fid].route_members:

                        roadLink_fid = data.roadRouteInformations[roadRouteInformation_fid].route_members['+']

                        orientation='pos'

                    else:

                        roadLink_fid = data.roadRouteInformations[roadRouteInformation_fid].route_members['-']

                        orientation='neg'

                    #Get the relevant terminal nodes so that we can look up the edge
                    v0 = data.roadLinks[roadLink_fid].tags['orientation_pos']
                    v1 = data.roadLinks[roadLink_fid].tags['orientation_neg']

                    #A try method is used here because sometimes the link to which
                    #the instruction refers is not actually in the graph (usually
                    #an edge effect) - the lookup would throw an error in such cases.
                    try:
                        g.edge[v0][v1][roadLink_fid]['one_way']=orientation
                    except:
                        pass

        self.g = g


    def build_posdict(self, data):
        '''
        Each node gets an attribute added for its geometric position. This is only
        really useful for plotting.
        '''

        for v in self.g:
            self.g.node[v]['loc'] = data.roadNodes[v[:-2]].eas_nor


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


def read_gml(filename):
    CurrentHandler=ITNHandler()
    sax.parse(filename, CurrentHandler)
    CurrentData = ITNData(CurrentHandler.roads,
                          CurrentHandler.roadNodes,
                          CurrentHandler.roadLinks,
                          CurrentHandler.roadLinkInformations,
                          CurrentHandler.roadRouteInformations)
    return CurrentData


if __name__ == '__main__':

    import os
    import numpy as np

    this_dir = os.path.dirname(os.path.realpath(__file__))
    ITNFILE = os.path.join(this_dir, 'test_data', 'mastermap-itn_417209_0_brixton_sample.gml')
    # ITNFILE = os.path.join(DATA_DIR, 'network_data/itn_sample', 'mastermap-itn_417209_0_brixton_sample.gml')

    # A little demo

    #Just build the network as usual
    itndata = read_gml(ITNFILE)
    g = ITNStreetNet.from_data_structure(itndata)
    grid_edge_index = g.build_grid_edge_index(50)

    # generate some random points inside camden
    xmin, ymin, xmax, ymax = g.extent
    grid_edge_idx = g.build_grid_edge_index(50)

    xs = np.random.rand(100)*(xmax - xmin) + xmin
    ys = np.random.rand(100)*(ymax - ymin) + ymin
    net_pts = [g.closest_edges_euclidean_brute_force(x, y)[1] for (x, y) in zip(xs, ys)]

    #Four test points - 1 and 3 on same segment, 2 on neighbouring segment, 4 long way away.
    test_points = [
        [531291, 175044],
        # [531293, 175054],
        [531185, 175207],
        [531466, 175005],
        [531643, 175061],
        [531724, 174826],
        [531013, 175294],
        [531426, 175315],
        [531459, 175075],
        [531007, 175037],
    ]
    source_points = []

    # Add these points as the kernel sources
    for i, t in enumerate(test_points):
        net_point, snap_distance = g.closest_edges_euclidean(t[0], t[1], grid_edge_index)
        source_points.append(net_point)
