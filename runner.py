__author__ = 'gabriel'
from django.db import connection
from database import models
import numpy as np
from django.contrib.gis import geos

def create_table():
    cursor = connection.cursor()
    cursor.execute(
        """CREATE TABLE crimes (
           pk integer);
           SELECT AddGeometryColumn('crimes', 'location', 3857, 'POINT', 2);"""
    );

def create_points(n=50):
    cursor = connection.cursor()
    cursor.execute("DELETE FROM crimes;")

    # arbitrary community area
    mpoly = models.ChicagoDivision.objects.get(type='chicago_community_area', name__iexact='bridgeport').mpoly
    # central side
    # ce = models.ChicagoDivision.objects.get(name='Central')
    # mpoly = ce.mpoly
    xmin, ymin, xmax, ymax = mpoly.extent

    # generate some random crimes
    xx = np.random.rand(n) * (xmax - xmin) + xmin
    yy = np.random.rand(n) * (ymax - ymin) + ymin
    pts = []

    for (x, y) in zip(xx, yy):
        tmp = geos.Point(x, y)
        tmp.srid = mpoly.srid
        tmp.transform(3857)
        sql = """INSERT INTO crimes (location) VALUES (ST_SetSRID(ST_MakePoint(%f, %f), 3857));""" % tmp.coords
        cursor.execute(sql)
