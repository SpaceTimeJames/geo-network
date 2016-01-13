__author__ = 'gabriel'
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.collections as mcoll
import matplotlib.path as mpath
from matplotlib import patches
from matplotlib import cm
import bisect
from descartes import PolygonPatch
from shapely import geometry, ops
from plotting.spatial import plot_shapely_geos
import os
import datetime

def colorline(
    x, y, z=None, cmap=plt.get_cmap('copper'), norm=None, ax=None,
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

    ax = ax if ax is None else plt.gca()
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
                            fmax=None,
                            vmax=None,
                            autoscale=True,
                            colorbar=True,
                            colorbar_values=False):

    # buffer all lines to polygon
    polys = [l.buffer(line_buffer) for l in lines]
    n = len(polys)

    if c is not None:
        assert not (fmax is not None and vmax is not None), "either specify fmax or vmax, not both"

        sort_idx = np.argsort(c)

        vmin = 0.
        if fmax:
            assert 0. < fmax <= 1., "fmax must be between 0 and 1"
            tmp = np.linspace(0, 1, len(lines))
            idx = bisect.bisect_left(tmp, fmax)
            vmax = np.array(c)[sort_idx][idx]
        elif vmax:
            pass
        else:
            vmax = max(c)

        norm = cm.colors.Normalize(vmin=vmin, vmax=vmax)
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array(c)
        patches = [
            PolygonPatch(polys[sort_idx[i]], edgecolor='none', facecolor=sm.to_rgba(c[sort_idx[i]]), alpha=alpha)
            for i in range(n)
        ]
    else:
        combined = ops.cascaded_union(polys)
        if not hasattr(combined, '__iter__'):
            combined = [combined]
        patches = [PolygonPatch(t, facecolor='none', edgecolor='k', linewidth=1, alpha=alpha) for t in combined]

    coll = mcoll.PatchCollection(patches, match_original=True)
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    ax.add_collection(coll)
    plt.draw()
    if autoscale:
        ax.set_aspect('auto')
    ax.set_aspect('equal')

    if c is not None and colorbar:
        fig = ax.get_figure()
        if colorbar_values:
            fig.colorbar(sm)
        else:
            fig.colorbar(sm, ticks=[])

    return coll


def plot_network_density(lines, edge_values, ax=None, fmax=0.99, **kwargs):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    plot_network_edge_lines(lines, c=edge_values, fmax=fmax, ax=ax, **kwargs)
    plot_network_edge_lines(lines, ax=ax, **kwargs)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()


def network_density_movie_slides(vb, res, t0=None, outdir='network_density_slides'):
    """
    Create image files of network density over a series of predictions
    :param vb: Validation object
    :param res: The result of a validation run
    :param t0: Optional datetime.date corresponding to time zero.
    :param outdir: The output directory for images, which is created if necessary
    :return: None
    """

    os.mkdir(outdir)
    n = len(res['cutoff_t'])
    idx = bisect.bisect_left(
        np.linspace(0, 1, res['prediction_values'].size),
        0.99
    )
    vmax = sorted(res['prediction_values'].flat)[idx]
    lines = list(vb.graph.lines_iter())

    for i in range(n):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
        ax.set_xticks([])
        ax.set_yticks([])

        plot_network_edge_lines(lines, c=res['prediction_values'][i], cmap=plt.get_cmap('copper'), vmax=vmax, ax=ax, line_buffer=20)
        # plot_network_edge_lines(lines, ax=ax, line_buffer=20)
        outfile = os.path.join(outdir, '%03d.png') % (i + 1)
        if t0:
            t = t0 + datetime.timedelta(days=res['cutoff_t'][i])
            title = t.strftime("%d/%m/%Y")
        else:
            title = res['cutoff_t'][i]
        plt.title(title, fontsize=24)
        plt.tight_layout(pad=1.5)
        plt.show()
        plt.axis('auto')
        plt.axis('equal')
        plt.savefig(outfile, dpi=150)
        plt.close(fig)


def network_density_movie_slides2(vb,
                                  res,
                                  t0=None,
                                  fmax=None,
                                  line_buffer=10,
                                  colorbar=False,
                                  boundary=None,
                                  outdir='network_density_slides'):
    """
    Create image files of network density over a series of predictions
    :param vb: Validation object
    :param res: The result of a validation run
    :param t0: Optional datetime.date corresponding to time zero.
    :param outdir: The output directory for images, which is created if necessary
    :param boundary: Optionally supply a Shapely object for plotting
    :param colorbar: TODO: needs to be implemented
    :return: None
    """

    os.mkdir(outdir)
    fmax = fmax or 0.99
    n = len(res['cutoff_t'])
    idx = bisect.bisect_left(
        np.linspace(0, 1, res['prediction_values'].size),
        fmax
    )
    vmax = sorted(res['prediction_values'].flat)[idx]
    lines = list(vb.graph.lines_iter())
    xy = vb.sample_points.to_cartesian()

    for i in range(n):
        z = vb.model.predict(res['cutoff_t'][i], None)  # no spatial points needed: reuse sample points
        z[z > vmax] = vmax
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.scatter(xy.toarray(0), xy.toarray(1), c=z, edgecolor='none', cmap='Reds')
        plot_network_edge_lines(lines, ax=ax, line_buffer=line_buffer)
        if boundary is not None:
            plot_shapely_geos(boundary, ax=ax, set_axes=False, facecolor='none', edgecolor='k', linewidth=2.)
        outfile = os.path.join(outdir, '%03d.png') % (i + 1)
        if t0:
            t = t0 + datetime.timedelta(days=res['cutoff_t'][i])
            title = t.strftime("%d/%m/%Y")
        else:
            title = res['cutoff_t'][i]
        plt.title(title, fontsize=24)
        plt.tight_layout(pad=1.5)
        plt.show()
        plt.axis('auto')
        plt.axis('equal')
        plt.savefig(outfile, dpi=150)
        plt.close(fig)


def network_lines_with_shaded_scatter_points(sample_points,
                                             prediction_value_arr,
                                             line_buffer=10,
                                             ax=None,
                                             fmin=None,
                                             fmax=None,
                                             vmin=None,
                                             vmax=None,
                                             boundary_poly=None,
                                             colourbar=False,
                                             cmap='Reds'):
    if vmax and fmax:
        raise AttributeError("Only supply one of vmax/fmax")
    if vmin and fmin:
        raise AttributeError("Only supply one of vmin/fmin")
    if fmax or fmin:
        ordered_vals = sorted(prediction_value_arr)
        if fmax:
            vmax = ordered_vals[int(fmax * len(ordered_vals))]
        if fmin:
            vmin = ordered_vals[int(fmin * len(ordered_vals))]
    if vmax:
        prediction_value_arr[prediction_value_arr > vmax] = vmax
    if vmin:
        prediction_value_arr[prediction_value_arr < vmin] = np.nan

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xticks([])
        ax.set_yticks([])

    xy = prediction_value_arr.to_cartesian()
    # create circular patches
    p = []
    for i in range(xy.ndata):
        p.append(patches.Circle(xy[i], radius=line_buffer, edgecolor=None))
    coll = mcoll.PatchCollection(p)
    coll.set_array(prediction_value_arr)
    coll.set_cmap(cmap)
    ax.add_collection(coll)

    # TODO: finish


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