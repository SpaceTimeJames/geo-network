__author__ = 'gabriel'
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.collections as mcoll
import matplotlib.path as mpath
from matplotlib import cm
import bisect
from descartes import PolygonPatch
from shapely import geometry, ops

def colorline(
    x, y, z=None, cmap=plt.get_cmap('copper'), norm=None,
        linewidth=3, alpha=1.0):
    """
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    """

    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))

    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])

    z = np.asarray(z)

    norm = norm or plt.Normalize(z.min(), z.max())

    segments = make_segments(x, y)
    lc = mcoll.LineCollection(segments, array=z, cmap=cmap, norm=norm,
                              linewidth=linewidth, alpha=alpha)

    ax = plt.gca()
    ax.add_collection(lc)

    return lc


def make_segments(x, y):
    """
    Create list of line segments from x and y coordinates, in the correct format
    for LineCollection: an array of the form numlines x (points per line) x 2 (x
    and y) array
    """

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments


def plot_network_edge_lines(lines,
                            c=None,
                            ax=None,
                            line_buffer=10,
                            alpha=0.75,
                            cmap=plt.get_cmap('Reds'),
                            fmax=1.0,
                            autoscale=True,
                            colorbar=True):

    # buffer all lines to polygon
    polys = [l.buffer(line_buffer) for l in lines]
    n = len(lines)

    if c is not None:
        assert 0. < fmax <= 1., "fmax must be between 0 and 1"
        tmp = np.linspace(0, 1, len(lines))
        idx = bisect.bisect_left(tmp, fmax)
        vmax = sorted(c)[idx]
        vmin = min(c)
        norm = cm.colors.Normalize(vmin=vmin, vmax=vmax)
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array(c)
        patches = [PolygonPatch(polys[i], edgecolor='none', facecolor=sm.to_rgba(c[i]), alpha=alpha) for i in range(n)]
    else:
        combined = ops.cascaded_union(polys)
        patches = [PolygonPatch(t, facecolor='none', edgecolor='k', linewidth=1, alpha=alpha) for t in combined]

    coll = mcoll.PatchCollection(patches, match_original=True)
    ax = ax or plt.gca()
    ax.add_collection(coll)
    plt.draw()
    if autoscale:
        ax.set_aspect('auto')
    ax.set_aspect('equal')

    if c is not None and colorbar:
        plt.colorbar(sm)

    return coll


def plot_network_density(lines, edge_values, fmax=0.99, **kwargs):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plot_network_edge_lines(lines, c=edge_values, fmax=fmax, ax=ax, **kwargs)
    plot_network_edge_lines(lines, ax=ax, **kwargs)
    ax.set_xticks([])
    ax.set_yticks([])


if __name__ == "__main__":
    N = 100
    np.random.seed(101)
    x = np.random.rand(N).cumsum()
    y = np.random.rand(N).cumsum()
    fig, ax = plt.subplots()

    path = mpath.Path(np.column_stack([x, y]))
    verts = path.interpolated(steps=3).vertices
    x, y = verts[:, 0], verts[:, 1]
    z = np.linspace(0, 10, len(x))
    colorline(x, y, z, cmap=plt.get_cmap('jet'), linewidth=2)

    plt.axis('auto')

    plt.show()