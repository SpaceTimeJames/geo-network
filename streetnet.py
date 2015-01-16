__author__ = 'Toby, refactor by Gabs'

from shapely.geometry import Point, LineString, Polygon
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

