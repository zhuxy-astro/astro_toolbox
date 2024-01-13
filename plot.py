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
matplotlib.rc('figure.subplot', bottom=0.125)  # to avoid text being covered
matplotlib.rc('savefig', format='pdf', dpi=300)


# %% wrapper: set_plot
# this takes only funcs with x and y, may have z. If the func has x only, use another func to wrap on it.
def set_plot(special_suffix=''):
    def decorate(drawing_func):
        @wraps(drawing_func)
        # TODO: set values for z_attr?
        @attr.set_values(
            set_default=True,
            to_set=[
                'x_label', 'x_left', 'x_right', 'x_line',
                'y_label', 'y_left', 'y_right', 'y_line',
            ])
        def set_plot_core(x, y, z=None,
                          select=slice(None),
                          plt_kwargs=dict(),
                          title='',
                          filename=None,  # 'X-Y-Z.pdf' by default. If set to '', no file will be saved.
                          savedir='figures',
                          fig_ax=None,  # if set, set as (fig, ax)
                          cmap='RdBu',  # if set to None, no cmap
                          proj=None,
                          **kwargs):
            """
            Draw the plot and save the file.

            Parameters
            ----------
            x, y, z: Columns or arrays.

            In kwargs:
                x_label=r'$X$', y_label=r'$Y$', z_label=r'$Z$',
                x_left=x_edge[0], x_right=x_edge[-1],
                y_left=y_edge[0], y_right=y_edge[-1],
                z_left=calc.min(z_data), z_right=calc.max(z_data),
                x_line=None, y_line=None,
            Returns
            -------
            fig, ax

            The wraped function:
                func(fig, ax, x, y, [z, ], [plt_kwargs=dict(),] **kwargs)
            """

            name_list = []
            name_list.append(attr.get_name(x, 'X'))
            name_list.append(attr.get_name(y, 'Y'))

            if z is not None:
                attr.choose_value(kwargs, 'z_left', z, 'left', calc.min, z)
                attr.choose_value(kwargs, 'z_right', z, 'right', calc.max, z)
                attr.choose_value(kwargs, 'z_label', z, 'label',
                                  attr.get_name(z, 'Z', to_latex=True))
                name_list.append(attr.get_name(z, 'Z'))

            if fig_ax is None:
                fig, ax = plt.subplots(subplot_kw={'projection': proj})
                ax.set_xlabel(kwargs['x_label'])
                ax.set_ylabel(kwargs['y_label'])
            else:
                # kwargs.pop('fig_ax')
                fig, ax = fig_ax
            if proj == 'aitoff':
                x = calc.to_radian(x, reset_zero=True)
                y = calc.to_radian(y, reset_zero=True)
            if proj == 'polar':
                x = calc.to_radian(x, reset_zero=True)

            select = attr.combine_selections(select)  # , reference=x)

            # avoid warnings (mostly UserWarning: The following kwargs were not used by func: ...)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # ---- DRAWING BEGINS----
                if z is None:
                    img = drawing_func(fig, ax,
                                       x, y,
                                       select=select,
                                       plt_kwargs=plt_kwargs,
                                       cmap=cmap,
                                       **kwargs)
                else:
                    img = drawing_func(fig, ax,
                                       x, y, z,
                                       select=select,
                                       plt_kwargs=plt_kwargs,
                                       cmap=cmap,
                                       **kwargs)
                # ---- DRAWING ENDS------

            if proj is None:
                ax.set_xlim(kwargs.get('x_left'), kwargs.get('x_right'))
                ax.set_ylim(kwargs.get('y_left'), kwargs.get('y_right'))
            elif proj == 'polar':
                ax.set_thetalim(calc.to_radian(kwargs.get('x_left')),
                                calc.to_radian(kwargs.get('x_right')))
                ax.set_rlim(kwargs.get('y_left'), kwargs.get('y_right'))

            already_has_cbar = fig.axes[-1].get_label() == '<colorbar>'
            if z is not None and cmap is not None and not already_has_cbar:
                cbar = fig.colorbar(img)
                img.set_clim(kwargs['z_left'], kwargs['z_right'])
                cbar.set_label(kwargs['z_label'])

            if kwargs['x_line'] is not None:
                ax.axvline(kwargs['x_line'], linewidth=2, alpha=0.5, c='g')
            if kwargs['y_line'] is not None:
                ax.axhline(kwargs['y_line'], linewidth=2, alpha=0.5, c='g')

            if (title == '') and hasattr(select, 'name'):
                title = select.name
            if title != '':
                ax.set_title(title, fontfamily='sans-serif', fontsize=16)

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
def img(fig, ax, x_edges, y_edges, z, plt_kwargs, bg=False, cmap='RdBu', **kwargs):
    """
    Parameters
    ----------
    x_edges, y_edges: Arrays of edges.
    z: 2-d binned map, or Column with data being the 2-d map.
    """
    img = ax.imshow(z,
                    extent=(x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]),
                    aspect='auto',
                    origin='lower',
                    vmin=kwargs['z_left'], vmax=kwargs['z_right'],
                    cmap=cmap,
                    **plt_kwargs)
    ax.grid()
    if bg:
        ax.set_facecolor('silver')
    return img


@set_plot()
def contour(fig, ax, x_edges, y_edges, z, plt_kwargs,
            contour_levels=15,
            cmap=None, **kwargs):
    """
    Parameters
    ----------
    x_edges, y_edges: Arrays of edges.
    z: 2-d binned map, or Column with data being the 2-d map.
    """
    # cmap and colors shouldn't be set at the same time
    if cmap is None:
        plt_kwargs.setdefault('colors', 'k')
    else:
        plt_kwargs.setdefault('cmap', cmap)
    plt_kwargs.setdefault('linewidths', 0.5)
    plt_kwargs.setdefault('linestyles', 'solid')

    x_centers = (x_edges[1:] + x_edges[:-1]) / 2
    y_centers = (y_edges[1:] + y_edges[:-1]) / 2
    x_centers_mesh, y_centers_mesh = np.meshgrid(x_centers, y_centers)
    img = ax.contour(x_centers_mesh, y_centers_mesh, z,
                     levels=np.linspace(kwargs['z_left'], kwargs['z_right'], contour_levels),
                     **plt_kwargs,
                     )
    return img


# %% func: plot scatter
@set_plot(special_suffix='scatter')
def scatter(fig, ax,
            x, y, z=None, select=slice(None),
            border=False,
            plt_kwargs=dict(),
            cmap='RdBu',
            s=10,  # markersize
            c=None,  # when z is set, c=z
            **kwargs):
    """
    Parameters
    ----------
    x, y[, z]: Arrays.
    """
    if z is not None:
        c = z[select]
        vmin = kwargs['z_left']
        vmax = kwargs['z_right']
    else:
        vmin = None
        vmax = None
        cmap = None

    if border:
        ax.scatter(x[select], y[select],
                   s=s, marker='o',
                   edgecolors='k',
                   lw=s / 20,
                   rasterized=True,
                   **plt_kwargs
                   )

    img = ax.scatter(x[select], y[select], c=c,
                     cmap=cmap,
                     s=s, lw=0, marker='o',
                     vmin=vmin, vmax=vmax,
                     rasterized=True,
                     # edgecolors='None'
                     **plt_kwargs
                     )
    """
    ax.scatter(radius_in_kpcold,gobs,marker='o',s=70, edgecolor='C9',alpha=0.8,
               label=r'$g_{obs}$',color='none')
    """
    ax.grid()

    return img


# %% func: bin_map
def bin_map(x, y, z=None,
            select=slice(None),
            plot_contour=1, plot_img=1,
            contour_kwargs=dict(), img_kwargs=dict(),
            **kwargs):
    """
    Binning x and y to calculate some function of z.
    Plot z, may with contour, or the contour of histogram.
    If z is not set, plot the 2-d histogram.

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
    z_map, x_edges, y_edges = calc.bin_map(x, y, z, select=select, **kwargs)
    x_edges = attr.array2column(x_edges, meta_from=x)
    y_edges = attr.array2column(y_edges, meta_from=y)

    if kwargs.get('plt_kwargs') is not None:
        raise ValueError('plt_kwargs is not allowed in bin_map. Use img_kwargs and contour_kwargs instead.')

    if z is None:
        z_map = attr.array2column(z_map, name='hist')
        if kwargs.get('weights') is None:
            attr.set(z_map, label='counts')
        else:
            attr.set(z_map, label='weighted counts')
        if plot_img:
            z_map_with_nan = attr.sift(z_map, min_=0, inplace=False)
            fig, ax = img(x_edges, y_edges, z_map_with_nan,
                          plt_kwargs=img_kwargs, select=select, **kwargs)
            kwargs['fig_ax'] = (fig, ax)
        if plot_contour:
            # kwargs['cmap'] = None
            fig, ax = contour(x_edges, y_edges, z_map,
                              plt_kwargs=contour_kwargs, select=select,
                              **kwargs)
    else:
        z_map = attr.array2column(z_map, meta_from=z)
        if plot_img:
            fig, ax = img(x_edges, y_edges, z_map,
                          plt_kwargs=img_kwargs, select=select, **kwargs)
            kwargs['fig_ax'] = (fig, ax)
        # kwargs['cmap'] = None
        if plot_contour == 1:
            fig, ax = contour(x_edges, y_edges, z_map,
                              plt_kwargs=contour_kwargs, select=select, **kwargs)
        elif plot_contour == 2:
            hist_map, x_edges, y_edges = calc.bin_map(x, y, **kwargs)
            fig, ax = contour(x_edges, y_edges, hist_map,
                              plt_kwargs=contour_kwargs, select=select, **kwargs)
    return fig, ax


# %% func: plot 1d hist
@set_plot()
def _bar(fig, ax, x, y, plt_kwargs=dict(), y_log=False, **kwargs):
    if y_log:
        ax.set_yscale('log')
    print(kwargs)
    img = ax.bar(x, y, width=x[1] - x[0], **plt_kwargs)
    return img


def hist(x,
         # select=slice(None),
         y_log=False,
         plt_kwargs=dict(),
         **kwargs):
    """
        x_label=r'$X$',
        y_left=0.,
        y_right=np.max(hist)*1.05
        y_label='count',
    """
    # select = attr.combine_selections(select, reference=x)
    x_center, h, h_err, kw = calc.hist(x, **kwargs)
    kwargs.update(kw)
    # attr.choose_value(kwargs, 'x_label', x, 'label', r'$X$')

    if y_log:
        kwargs['y_left'] = kwargs.get('y_left', 10.)
        kwargs['y_right'] = kwargs.get('y_right', calc.max(h) * 1.1)
    else:
        kwargs['y_left'] = kwargs.get('y_left', 0.)
        kwargs['y_right'] = kwargs.get('y_right', calc.max(h) * 1.05)

    if kwargs.get('weights') is None:
        kwargs['y_label'] = 'count'
    else:
        kwargs['y_label'] = 'weighted count'

    x_center = attr.array2column(x_center, meta_from=x)
    h = attr.array2column(h, name='hist')

    fig, ax = _bar(x_center, h,
                   # select=select,
                   plt_kwargs=plt_kwargs,
                   **kwargs)
    ax.errorbar(x_center, h, yerr=h_err, ls='', markersize=0, elinewidth=1.5, ecolor='k')
    return fig, ax


# %% func: plot mean with std
@set_plot()
def errorbar(fig, ax, x_edges, y_means, y_err=None,
             marker='o', markersize=7,
             c_err='tab:blue',
             plt_kwargs=dict(), **kwargs):
    """
    This is to plot mean with std, not for individual points.
    """
    # y_err = kwargs.get('y_err')
    x_centers = (x_edges[1:] + x_edges[:-1]) / 2
    img = ax.errorbar(x_centers, y_means, yerr=y_err,
                      marker=marker, markersize=markersize,
                      c=c_err, **plt_kwargs)
    return img


def bin_x(x, y, plot_scatter=False, func=None,
          select=slice(None),
          errorbar_kwargs=dict(), scatter_kwargs=dict(),
          **kwargs):

    if kwargs.get('plt_kwargs') is not None:
        raise ValueError('plt_kwargs is not allowed in bin_map. Use img_kwargs and contour_kwargs instead.')

    select = attr.combine_selections(select, reference=x)
    x = x[select]
    y = y[select]
    x_left = attr.choose_value(kwargs, 'x_left', x, 'left', calc.min, x)
    x_right = attr.choose_value(kwargs, 'x_right', x, 'right', calc.max, x)
    nbins = kwargs.get('bins', 10)

    x_edges = np.linspace(x_left, x_right, nbins + 1)
    digitized_x = np.digitize(x, x_edges)
    binned_y = [y[digitized_x == i] for i in range(1, nbins + 1)]
    if func is None:
        ys = np.array([calc.mean(y_in_bin) for y_in_bin in binned_y])
        y_std = np.array([np.nanstd(y_in_bin) for y_in_bin in binned_y])
    else:
        ys = np.array([func(y_in_bin) for y_in_bin in binned_y])
        y_std = None

    x_edges = attr.array2column(x_edges, meta_from=x)
    ys = attr.array2column(ys, meta_from=y)

    fig, ax = errorbar(x_edges, ys, y_err=y_std, plt_kwargs=errorbar_kwargs,
                       **kwargs)
    kwargs.pop('fig_ax', None)

    if plot_scatter:
        c = kwargs.pop('c', 'silver')
        fig, ax = scatter(x, y, z=None, fig_ax=(fig, ax),
                          c=c, plt_kwargs=scatter_kwargs,
                          **kwargs)

    return fig, ax


# %% func: loess2d
@set_plot(special_suffix='loess')
def loess(fig, ax, x, y, z, plt_kwargs, border=True,
          select=slice(None),
          cmap='RdBu',
          **kwargs):
    """
    kwargs: s=80
    """
    s = kwargs.get('s', 80)
    # TODO: if select is slice(None) ...
    for i in [x, y, z]:
        select &= np.isfinite(i)
    x_use = x[select]
    y_use = y[select]
    z_use = z[select]
    use_kai = 0
    if use_kai:
        loess = calc.LOESS2D(x_use, y_use, z_use, 30)
        zout = loess(x_use, y_use)
    else:
        # written by Capperllari, super slow for large set of data!
        from loess import loess_2d
        zout, wout = loess_2d.loess_2d(x_use, y_use, z_use)  # , npoints=30)
    # plot the bg shadow first, then plot the colors
    if border:
        ax.scatter(x_use, y_use, edgecolors='k', lw=4.5, s=s, rasterized=True, **plt_kwargs)
    img = ax.scatter(x_use, y_use, c=zout, cmap=cmap, lw=0, s=s, rasterized=True, **plt_kwargs)
    return img
