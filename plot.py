#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# %% import
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from functools import wraps
import os
import warnings
# from pandas import DataFrame, Series
# from astropy.table import Table
from . import attr
# import astro_toolbox.calc as calc
from . import calc

# %% settings
# settings here will directly affect the file importing this module.

# from importlib import reload
# reload(plt)
# reload(matplotlib)

# Yangyao Chen's settings
# plt.style.use(["classic"])
# matplotlib.rc('font', family='monospace', weight='normal', size=20)
matplotlib.rc('lines', linewidth=3)

c_frame = (0, 0, 0, .8)
for tick in 'xtick', 'ytick':
    matplotlib.rc(f'{tick}.major', width=1.5, size=8)
    matplotlib.rc(f'{tick}.minor', width=1, size=4, visible=True)
    matplotlib.rc(tick, color=c_frame, labelsize=15, direction='in')
matplotlib.rc('xtick', top=True)
matplotlib.rc('ytick', right=True)
matplotlib.rc('axes', linewidth=1.5, edgecolor=c_frame, labelweight='normal')
# matplotlib.rc('grid', color=c_frame)
matplotlib.rc('patch', edgecolor=c_frame)

# my own settings
# matplotlib.rc('font', family='serif', serif="Times New Roman", size=18)
matplotlib.rc('font', size=18)
matplotlib.rc('mathtext', fontset='cm')
matplotlib.rc('figure.subplot', bottom=0.125, left=0.14)  # to avoid text being covered
matplotlib.rc('savefig', format='pdf', dpi=300)


# %% wrapper: set_title_filename
# TODO: decorate set_plot, bin_map, contour_scatter

# %% wrapper: set_plot
def set_plot(special_suffix=''):
    def decorate(drawing_func):
        @wraps(drawing_func)
        @attr.set_values(
            set_default=True,
            to_set=['x_label', 'x_left', 'x_right', 'x_line',
                    'y_label', 'y_left', 'y_right', 'y_line',
                    'z_label', 'z_left', 'z_right', 'z_line'])
        def set_plot_core(x, y=None, z=None,
                          select=slice(None),
                          plt_args=dict(),
                          title='',
                          filename=None, savedir='figures',
                          fig_ax=None,
                          cbar=True,
                          proj=None,
                          legend=12,
                          **kwargs):
            """
            Draw the plot and save the file.

            Parameters
            ----------
            x, y=None, z=None: Columns or arrays.
            select=slice(None),
            plt_args=dict(),
            title='',  # title of the plot, the name of the selections by default.
            filename=None,  # 'X-Y-Z-{special_suffix}, {title}.pdf' by default. If set to '', no file will be saved.
            savedir='figures',
            fig_ax=None,  # if set, set as (fig, ax)
            cbar=True,
            proj=None,  # could be 'aitoff' or 'polar'
            legend=12,  # 0: no legend, >0: legend fontsize

            In kwargs:
                'x_label', 'x_left', 'x_right', 'x_line',
                'y_label', 'y_left', 'y_right', 'y_line',
                'z_label', 'z_left', 'z_right', 'z_line'

            Returns
            -------
            fig, ax

            The wraped function:
                func(fig, ax, x, [y, z,] [plt_args=dict(),] **kwargs)
            """
            have_y = y is not None
            have_z = z is not None

            name_list = []
            name_list.append(attr.get_name(x, 'X'))
            if have_y:
                name_list.append(attr.get_name(y, 'Y'))
            if have_z:
                name_list.append(attr.get_name(z, 'Z'))

            if fig_ax is None:
                fig, ax = plt.subplots(subplot_kw={'projection': proj})
            else:
                fig, ax = fig_ax

            ax.set_xlabel(kwargs['x_label'])
            if have_y:
                ax.set_ylabel(kwargs['y_label'])

            if have_y and proj == 'aitoff':
                x = calc.to_radian(x, reset_zero=True)
                y = calc.to_radian(y, reset_zero=True)
            if proj == 'polar':
                x = calc.to_radian(x, reset_zero=True)

            # no reference here is set because select and x may have different dimensions.
            # In such case only the combined name is used.
            select = attr.combine_selections(select)  # , reference=x)

            # avoid warnings (mostly UserWarning: The following kwargs were not used by func: ...)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # ---- DRAWING BEGINS----
                x_y_z = [x]
                if have_y:
                    x_y_z += [y]
                if have_z:
                    x_y_z += [z]
                img = drawing_func(fig, ax,
                                   *x_y_z,
                                   select=select,
                                   plt_args=plt_args,
                                   **kwargs)
                # ---- DRAWING ENDS------

            if proj is None:
                ax.set_xlim(kwargs.get('x_left'), kwargs.get('x_right'))
                if have_y:
                    ax.set_ylim(kwargs.get('y_left'), kwargs.get('y_right'))
            elif have_y and proj == 'polar':
                ax.set_thetalim(calc.to_radian(kwargs.get('x_left')),
                                calc.to_radian(kwargs.get('x_right')))
                ax.set_rlim(kwargs.get('y_left'), kwargs.get('y_right'))

            already_has_cbar = fig.axes[-1].get_label() == '<colorbar>'
            if have_z and cbar and not already_has_cbar:
                cbar = fig.colorbar(img)
                img.set_clim(kwargs['z_left'], kwargs['z_right'])
                cbar.set_label(kwargs['z_label'])

            if kwargs['x_line'] is not None:
                ax.axvline(kwargs['x_line'], linewidth=2, alpha=0.5, c='g')
            if have_y and kwargs['y_line'] is not None:
                ax.axhline(kwargs['y_line'], linewidth=2, alpha=0.5, c='g')

            if (title == '') and hasattr(select, 'name'):
                title = select.name
            if title != '':
                ax.set_title(title, fontfamily='sans-serif', fontsize=16)

            if legend and len(ax.get_legend_handles_labels()[0]) > 0:
                ax.legend(fontsize=legend)

            if filename is None:
                if special_suffix != '':
                    name_list += [special_suffix]
                filename = '-'.join(name_list)
                if title != '':
                    filename = f'{filename}, {title}'
                filename = f'{filename}.pdf'

            if not os.path.exists(savedir):
                os.mkdir(savedir)
            if filename != '':
                fig.savefig(os.path.join(savedir, filename))

            fig.show()
            return fig, ax
        return set_plot_core
    return decorate


# %% func: img and contour
@set_plot()
def img(fig, ax, x_edges, y_edges, z, plt_args, plot_bg=False, **kwargs):
    """
    Parameters
    ----------
    x_edges, y_edges: Arrays of edges, NOT the centers. The dim of edges should be len(z) + 1.
    z: 2-d binned map, or Column with data being the 2-d map.
    """
    plt_args.setdefault('cmap', 'RdBu')
    img = ax.imshow(z,
                    extent=(x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]),
                    aspect='auto',
                    origin='lower',
                    vmin=kwargs['z_left'], vmax=kwargs['z_right'],
                    **plt_args)
    ax.grid()
    if plot_bg:
        ax.set_facecolor('silver')
    return img


@set_plot()
def contour(fig, ax, x_edges, y_edges, z, plt_args,
            contour_levels=15,
            **kwargs):
    """
    Parameters
    ----------
    x_edges, y_edges: Arrays of edges, NOT the centers. The dim of edges should be len(z) + 1.
    z: 2-d binned map, or Column with data being the 2-d map.
    """
    default_plt_args = {
        'colors': 'k',
        'linewidths': 0.5,
        'linestyles': 'solid'
    }
    # cmap and colors shouldn't be set at the same time
    # use colors by default, unless cmap is set
    if plt_args.get('cmap') is None:
        plt_args.setdefault('colors', default_plt_args.pop('colors'))
    # else plt_args has cmap
    for k, v in default_plt_args.items():
        plt_args.setdefault(k, v)

    x_centers = (x_edges[1:] + x_edges[:-1]) / 2
    y_centers = (y_edges[1:] + y_edges[:-1]) / 2
    x_centers_mesh, y_centers_mesh = np.meshgrid(x_centers, y_centers)
    img = ax.contour(x_centers_mesh, y_centers_mesh, z,
                     levels=np.linspace(kwargs['z_left'], kwargs['z_right'], contour_levels),
                     **plt_args,
                     )
    return img


# %% func: plot scatter
@set_plot(special_suffix='scatter')
def scatter(fig, ax,
            x, y, z=None, select=slice(None),
            plot_border=False,
            plt_args=dict(),
            **kwargs):
    """
    Parameters
    ----------
    x, y[, z]: Arrays.
    """
    default_plt_args = dict(
        marker='o',
        s=10,  # marker size
        rasterized=True,
        cmap='RdBu'
    )
    for k, v in default_plt_args.items():
        plt_args.setdefault(k, v)

    # plot the bg shadow first, then plot the colors
    if plot_border:
        border_args = plt_args.copy()
        border_args.pop('cmap', None)  # no cmap for border
        border_args.setdefault('edgecolors', 'k')
        border_args.setdefault('lw', border_args['s'] / 20)
        ax.scatter(x[select], y[select],
                   **border_args
                   )

    if z is not None:
        c = z[select]
        vmin = kwargs['z_left']
        vmax = kwargs['z_right']
    else:
        c = plt_args.pop('c', None)
        vmin = None
        vmax = None
        plt_args['cmap'] = None

    plt_args.setdefault('lw', 0)
    img = ax.scatter(x[select], y[select],
                     c=c,  # c is z, or None, or set manually
                     vmin=vmin, vmax=vmax,
                     # edgecolors='None'
                     **plt_args
                     )
    ax.grid()

    return img


# %% func: bin_map
def bin_map(x, y, z=None,
            select=slice(None),
            plot_contour=1, plot_img=1,
            z_log=False,
            step_follow_window=False,
            contour_args=dict(), img_args=dict(),
            **kwargs):
    """
    Binning x and y to calculate some function of z.
    Plot z, may with contour, or the contour of histogram.
    If z is not set, plot the 2-d histogram.
    weights is within kwargs and passed to calc.bin_map
    Need to set filenames manually.

    In TOPCAT, the color in "auto" and "density" is plotted based on the number of layers of markers on each pixel.
    If treating the `step`s here as pixels and set square markers with size of `window`s,
    this is exactly what is done here.
    The window size controls both the smoothness in the high density regions and the marker size in outliers,
    while steps controls the resolution in high density regions, and if the outliers are grid-like.

    Parameters
    ----------
    x, y, [z]: Column or simply 1-d array of data.
    plot_contour=1,  # 0: no contour, 1: contour of z, 2: contour of histogram
    plot_img=1,  # 0: no img, 1: img of z, 2: img of histogram

    In kwargs:
        everything of calc.bin_map, img and contour

    Returns
    -------
    fig, ax
    """
    select = attr.combine_selections(select, reference=x)
    z_map, x_edges, y_edges = calc.bin_map(x, y, z, select=select,
                                           step_follow_window=step_follow_window, **kwargs)
    if z_log:
        z_map = np.log10(z_map)
        kwargs['cbar'] = False
    if z is None and not step_follow_window:
        # calc histogram, but not the real number in each window
        kwargs['cbar'] = False
    x_edges = attr.array2column(x_edges, meta_from=x)
    y_edges = attr.array2column(y_edges, meta_from=y)
    # edges is left here not converted into centers because this is done in scatter and img.

    if kwargs.get('plt_args') is not None:
        raise ValueError('plt_args is not allowed in bin_map. Use img_args and contour_args instead.')

    if z is None:
        z_map = attr.array2column(z_map, name='hist')
        if kwargs.get('weights') is None:
            attr.set(z_map, label='counts')
        else:
            attr.set(z_map, label='weighted counts')
        if plot_img:
            z_min_ = -int(z_log)  # -1 if z is in log scale, otherwise 0
            z_map_with_nan = attr.sift(z_map, min_=z_min_, inplace=False)
            fig, ax = img(x_edges, y_edges, z_map_with_nan,
                          plt_args=img_args, select=select, **kwargs)
            kwargs['fig_ax'] = (fig, ax)
        if plot_contour:
            # kwargs['cmap'] = None
            fig, ax = contour(x_edges, y_edges, z_map,
                              plt_args=contour_args, select=select,
                              **kwargs)
    else:
        z_map = attr.array2column(z_map, meta_from=z)
        if plot_img:
            fig, ax = img(x_edges, y_edges, z_map,
                          plt_args=img_args, select=select, **kwargs)
            kwargs['fig_ax'] = (fig, ax)
        # kwargs['cmap'] = None
        if plot_contour == 1:
            fig, ax = contour(x_edges, y_edges, z_map,
                              plt_args=contour_args, select=select, **kwargs)
        elif plot_contour == 2:
            hist_map, x_edges, y_edges = calc.bin_map(x, y, **kwargs)
            fig, ax = contour(x_edges, y_edges, hist_map,
                              plt_args=contour_args, select=select, **kwargs)
    return fig, ax


# %% func: contour_scatter
def contour_scatter(x, y,
                    select=slice(None),
                    contour_args=dict(), scatter_args=dict(),
                    percentile=0.99,
                    **kwargs):
    """
    `percentile` could be a number or an array. If a number, a set of 20 levels is used.
    `plt.contour` is not good at dealing with sharp edges.
    In kwargs:
        everything of calc.bin_map and scatter. `weights` is within kwargs and passed to calc.bin_map.
    Need to set filenames manually.
    """
    if kwargs.get('plt_args') is not None:
        raise ValueError('plt_args is not allowed in contour_scatter. Use scatter_args and contour_args instead.')

    default_scatter_args = dict(
        s=10,
        rasterized=True,
        c='cornflowerblue',
        alpha=0.5,
        lw=0)
    for k, v in default_scatter_args.items():
        scatter_args.setdefault(k, v)
    fig, ax = scatter(x, y,
                      plt_args=scatter_args, select=select,
                      filename='',
                      **kwargs)

    z_map, x_edges, y_edges = calc.bin_map(x, y, select=select, **kwargs)
    z_map /= np.nansum(z_map)
    x_edges = attr.array2column(x_edges, meta_from=x)
    y_edges = attr.array2column(y_edges, meta_from=y)

    z_map = attr.array2column(z_map, name='contour_scatter')
    percentile_out = 1 - np.array(percentile)
    try:
        iter(percentile)
        # percentile is a list
        hist_threshold = calc.weighted_percentile(weights=z_map, percentile=percentile_out)
        hist_threshold = np.sort(hist_threshold)
        # list here is to make sure the max is appended, not added.
        if hist_threshold[-1] < z_map.max():
            hist_threshold = np.append(np.array(hist_threshold), z_map.max())
    except TypeError:
        hist_threshold = calc.weighted_percentile(weights=z_map, percentile=percentile_out)
        # hist_threshold = calc.hist_percentile(z_map, [percentile])
        hist_threshold = np.linspace(hist_threshold, z_map.max(), 20)

    x_centers = (x_edges[1:] + x_edges[:-1]) / 2
    y_centers = (y_edges[1:] + y_edges[:-1]) / 2
    contour_args.setdefault('cmap', 'coolwarm')
    ax.contourf(x_centers, y_centers, z_map,
                levels=hist_threshold,
                **contour_args)
    return fig, ax


# %% func: plot 1d hist
@set_plot()
def _bar(fig, ax, x, y, plt_args, y_log=False, **kwargs):
    if y_log:
        ax.set_yscale('log')
    img = ax.bar(x, y, width=x[1] - x[0], **plt_args)
    return img


def hist(x,
         y_log=False,
         plot_errorbar=True,
         bar_args=dict(),
         errorbar_args=dict(),
         **kwargs):
    """
    kwargs: everything in calc.hist
    """

    if kwargs.get('plt_args') is not None:
        raise ValueError('plt_args is not allowed in hist. Use errorbar_args and bar_args instead.')

    h, h_err, x_center, kw = calc.hist(x, **kwargs)
    kwargs.update(kw)

    if y_log:
        kwargs['y_left'] = kwargs.get('y_left', 10.)
        kwargs['y_right'] = kwargs.get('y_right', calc.max(h) * 1.1)
    else:
        kwargs['y_left'] = kwargs.get('y_left', 0.)
        kwargs['y_right'] = kwargs.get('y_right', calc.max(h) * 1.05)

    if kwargs.get('norm', False):
        kwargs['y_label'] = 'PDF'
    elif kwargs.get('weights') is None:
        kwargs['y_label'] = 'count'
    else:
        kwargs['y_label'] = 'weighted count'

    x_center = attr.array2column(x_center, meta_from=x)
    h = attr.array2column(h, name='hist')

    fig, ax = _bar(x_center, h,
                   plt_args=bar_args,
                   **kwargs)
    if plot_errorbar:
        default_errorbar_args = dict(
            elinewidth=1.5,
            ecolor='k',
            markersize=0,
            ls='',
        )
        for k, v in default_errorbar_args.items():
            errorbar_args.setdefault(k, v)
        kwargs.pop('fig_ax', None)
        fig, ax = errorbar(x_center, h, y_err=h_err, fig_ax=(fig, ax),
                           plt_args=errorbar_args,
                           **kwargs)

    return fig, ax


# %% func: plot mean with std
@set_plot()
def errorbar(fig, ax, x_centers, y_means, y_err=None,
             plt_args=dict(), **kwargs):
    """
    This is to plot mean with std, not for individual points.
    """

    default_plt_args = dict(
        marker='o', markersize=7,
        c='tab:blue',
    )
    for k, v in default_plt_args.items():
        plt_args.setdefault(k, v)

    img = ax.errorbar(x_centers, y_means, yerr=y_err, **plt_args)
    return img


def bin_x(x, y=None, y_log=False, mode='mean',
          plot_scatter=False, plot_errorbar=True, plot_fill=True,
          errorbar_args=dict(), scatter_args=dict(), fill_args=dict(), bar_args=dict(),
          **kwargs):
    """
    If errorbar is not set, plot the filled area.
    kwargs: everything in calc.bin_x
    """

    if kwargs.get('plt_args') is not None:
        raise ValueError('plt_args is not allowed in bin_x. Use errorbar_args and scatter_args instead.')

    ys, y_err, x_centers = calc.bin_x(x, y, **kwargs)
    ys = attr.array2column(ys, meta_from=y)

    x_centers = attr.array2column(x_centers, meta_from=x)

    if plot_errorbar:
        y_err_used = y_err
    else:
        y_err_used = None
        errorbar_args.update({'markersize': 0})
    fig, ax = errorbar(x_centers, ys, y_err=y_err_used, plt_args=errorbar_args, **kwargs)

    if not plot_errorbar and plot_fill:
        fill_args.setdefault('alpha', 0.2)
        if mode == 'median':
            ax.fill_between(x_centers, ys - y_err[0], ys + y_err[1], **fill_args)
        else:
            ax.fill_between(x_centers, ys - y_err, ys + y_err, **fill_args)
    kwargs.pop('fig_ax', None)

    if y is not None and plot_scatter:
        scatter_args['c'] = 'silver'  # silver scatter in the background
        fig, ax = scatter(x, y, z=None, fig_ax=(fig, ax),
                          plt_args=scatter_args,
                          **kwargs)

    return fig, ax


# %% func: loess2d
@set_plot(special_suffix='loess')
def loess(fig, ax, x, y, z, plt_args, plot_border=True,
          select=slice(None),
          **kwargs):
    select = attr.combine_selections(select, reference=x)
    for i in [x, y, z]:
        select &= calc.select_good(i)
    x_use = x[select]
    y_use = y[select]
    z_use = z[select]

    use_kai = 1
    if use_kai:
        loess = calc.LOESS2D(x_use, y_use, z_use, 30)
        zout = loess(x_use, y_use)
    else:
        # written by Capperllari, super slow for large set of data!
        from loess import loess_2d
        zout, wout = loess_2d.loess_2d(x_use, y_use, z_use)  # , npoints=30)

    default_plt_args = dict(
        s=80,
        rasterized=True,
        cmap='RdBu',
    )
    for k, v in default_plt_args.items():
        plt_args.setdefault(k, v)

    # plot the bg shadow first, then plot the colors
    if plot_border:
        border_args = plt_args.copy()
        border_args.pop('cmap', None)
        border_args.setdefault('edgecolors', 'k')
        # border_args.setdefault('lw', border_args['s'] / 20)
        border_args.setdefault('lw', 4.5)
        ax.scatter(x_use, y_use, **border_args)

    img = ax.scatter(x_use, y_use, c=zout, lw=0, **plt_args)
    return img
