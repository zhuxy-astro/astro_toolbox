#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# %% import
from functools import wraps
import os
import warnings

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# from pandas import DataFrame, Series
# from astropy.table import Table
# import astro_toolbox.calc as calc

from . import attr, calc, sel

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
matplotlib.rc('figure.subplot', bottom=0.125, left=0.2)  # to avoid labels being covered by the window boundary
matplotlib.rc('axes', labelpad=1.)  # make the labels closer to the axis, by default 4.0
matplotlib.rc('savefig', format='pdf', dpi=300, directory='~/Downloads')

default_cmap = 'coolwarm'

default_savedir = 'figures'


# %% func: set_title_save_fig
def set_title_save_fig(ax, x, y=None, z=None, savedir=default_savedir,
                       special_suffix='', select=[],
                       filename=None, title=None,
                       filetype='pdf'
                       ):
    """
    save file as 'figures/X-Y-Z-{special_suffix}, {title}.{filetype}' by default.

    x, y, z: Columns or str.

    title = None: use the name of the selections by default, no title if ''.
        only used as title above the figure, not in the filename

    If filename is set to '', no file will be saved.
    Used by plotting functions not decorated by set_plot.
    Requirement for the definition of funcs using this func:
        select=slice(None), savedir=default_savedir, filename=None

    special_suffix is usually used for different plot styles, for example 'contour'.
    """
    if filename == '':
        return

    if filename is None:
        name_list = []
        name_list.append(attr.get_name(x, 'X'))
        if y is not None:
            name_list.append(attr.get_name(y, 'Y'))
        if z is not None:
            name_list.append(attr.get_name(z, 'Z'))

        if special_suffix != '':
            name_list += [special_suffix]
        filename = '-'.join(name_list)

        select_name = sel.combine_names(select)
        if select_name != '':
            filename = f'{filename}, {select_name}'
    # else, use the overwritten filename

    filename = f'{filename}.{filetype}'

    if title is None and select_name != '':
        title = select_name
    # if title != '':
    # set title even it is empty
    ax.set_title(title, fontfamily='sans-serif', fontsize=16)

    if not os.path.exists(savedir):
        os.mkdir(savedir)

    ax.figure.savefig(os.path.join(savedir, filename))


# %% wrapper: set_plot
def set_plot(special_suffix=''):
    """special_suffix is usually used for different plot styles, for example 'contour'.
    """
    def decorate(drawing_func):
        @wraps(drawing_func)
        @attr.set_values(
            set_default=True,
            to_set=['x_label', 'x_left', 'x_right', 'x_line',
                    'y_label', 'y_left', 'y_right', 'y_line',
                    'z_label', 'z_left', 'z_right', 'z_line', 'z_cmap'])
        def set_plot_core(x, y=None, z=None, *,
                          select=slice(None),
                          plt_args=None,
                          title=None,
                          filename=None, savedir=default_savedir,
                          ax=None, cbar_ax=None,
                          plot_cbar=True,
                          plot_bg=False,
                          proj=None,
                          legend=12,
                          **kwargs):
            """
            Draw the plot and save the file.

            Parameters
            ----------
            x, y=None, z=None: Columns or arrays.
            select=slice(None),
            plt_args=None,  # automatically set to dict(). No need to set manually in the wrapped function.
            title=None,
            filename=None,
            savedir=default_savedir,
            ax=None,  # if set, set as ax
            plot_cbar=True,
            proj=None,  # could be 'aitoff' or 'polar'
            legend=12,  # 0: no legend, >0: legend fontsize

            In kwargs:
                'x_label', 'x_left', 'x_right', 'x_line',
                'y_label', 'y_left', 'y_right', 'y_line',
                'z_label', 'z_left', 'z_right', 'z_line', 'z_cmap'

            Returns
            -------
            ax

            The wraped function:
                func(ax, x, [y, z,] [plt_args=dict(),] **kwargs)
            """
            have_y = y is not None
            have_z = z is not None

            if plt_args is None:
                plt_args = dict()

            if ax is None:
                fig, ax = plt.subplots(subplot_kw={'projection': proj})
            else:
                fig = ax.figure

            ax.set_xlabel(kwargs['x_label'])
            if have_y:
                ax.set_ylabel(kwargs['y_label'])

            if have_y and proj == 'aitoff':
                x = calc.deg_to_radian(x, reset_zero=True)
                y = calc.deg_to_radian(y, reset_zero=True)
            if proj == 'polar':
                x = calc.deg_to_radian(x, reset_zero=True)

            # no reference here is set because select and x may have different dimensions.
            # In such case only the combined name is used.
            select = sel.combine(select)  # , reference=x)

            # avoid warnings (mostly UserWarning: The following kwargs were not used by func: ...)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # ---- DRAWING BEGINS----
                x_y_z = [x]
                if have_y:
                    x_y_z += [y]
                if have_z:
                    x_y_z += [z]
                img = drawing_func(ax,
                                   *x_y_z,
                                   select=select,
                                   plt_args=plt_args,
                                   cbar_ax=cbar_ax,
                                   **kwargs)
                # ---- DRAWING ENDS------

            if proj is None:
                ax.set_xlim(kwargs.get('x_left'), kwargs.get('x_right'))
                if have_y:
                    ax.set_ylim(kwargs.get('y_left'), kwargs.get('y_right'))
            elif have_y and proj == 'polar':
                ax.set_thetalim(calc.deg_to_radian(kwargs.get('x_left')),
                                calc.deg_to_radian(kwargs.get('x_right')))
                ax.set_rlim(kwargs.get('y_left'), kwargs.get('y_right'))

            already_has_cbar = fig.axes[-1].get_label() == '<colorbar>'
            already_has_cbar &= cbar_ax is None
            if have_z and plot_cbar and not already_has_cbar:
                cbar = fig.colorbar(img, ax=ax, cax=cbar_ax)
                img.set_clim(kwargs['z_left'], kwargs['z_right'])
                cbar.set_label(kwargs['z_label'])

            if kwargs['x_line'] is not None:
                ax.axvline(kwargs['x_line'], linewidth=2, alpha=0.5, c='g')
            if have_y and kwargs['y_line'] is not None:
                ax.axhline(kwargs['y_line'], linewidth=2, alpha=0.5, c='g')

            if plot_bg:
                ax.set_facecolor('silver')

            if legend and len(ax.get_legend_handles_labels()[0]) > 0:
                ax.legend(fontsize=legend)

            set_title_save_fig(ax=ax, x=x, y=y, z=z, title=title,
                               savedir=savedir, special_suffix=special_suffix,
                               select=select, filename=filename)

            fig.show()
            return ax
        return set_plot_core
    return decorate


# %% func: img and contour
@set_plot()
def img(ax, x_edges, y_edges, z, plt_args, **kwargs):
    """
    Parameters
    ----------
    x_edges, y_edges: Arrays of edges, NOT the centers. The dim of edges should be len(z) + 1.
    z: 2-d binned map, or Column with data being the 2-d map.
    """
    # kwargs['z_cmap'] is set to None by default, so that methods like `setdefault` or `kwargs.get` don't work.
    if kwargs.get('z_cmap') is None:
        kwargs['z_cmap'] = default_cmap

    default_plt_args = dict(
        extent=(x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]),
        aspect='auto',
        origin='lower',
        vmin=kwargs['z_left'], vmax=kwargs['z_right'],
        cmap=kwargs['z_cmap'],
    )

    if plt_args is not None:
        default_plt_args.update(plt_args)
    plt_args = default_plt_args

    img = ax.imshow(z, **plt_args)
    ax.grid()
    return img


def _calc_contour_levels(z, contour_levels, **kwargs):
    """
    contour_levels:
        if int, it is the number of levels;
        if a float between 0 and 1, it is the percentile, and z is treated as normalized histogram. 15 levels is used;
        if a list of floats all between 0 and 1, it is the list of percentiles, with z treated as normalized histogram;
        if a list of floats not all between 0 and 1, it is the list of levels of z.
    """
    # handle contour_levels
    levels_is_list = np.ndim(contour_levels) > 0

    if not levels_is_list and isinstance(contour_levels, int):
        assert contour_levels > 0, 'contour_levels should be a positive integer.'
        levels = np.linspace(kwargs['z_left'], kwargs['z_right'], contour_levels)

    elif not levels_is_list and isinstance(contour_levels, float):
        assert 0 < contour_levels < 1, 'contour_levels should be a float between 0 and 1.'
        percentile_outside = 1 - contour_levels
        levels = calc.weighted_percentile(weights=z, percentile=percentile_outside)
        levels = np.linspace(levels, z.max(), 15)

    elif levels_is_list and all([0 <= i <= 1 for i in contour_levels]):
        percentile_outside = 1 - np.array(contour_levels)
        levels = calc.weighted_percentile(weights=z, percentile=percentile_outside)
        levels = np.sort(levels)
        if levels[-1] < z.max():
            levels = np.append(np.array(levels), z.max())

    elif levels_is_list:
        levels = contour_levels

    else:
        raise ValueError('contour_levels should be an int, a float between 0 and 1, or a list.')

    return levels


@set_plot()
def _contour(ax, x_edges, y_edges, z, plt_args,
             plot_contourf=False,
             contour_levels=15,
             labels=None,
             clabel_args=None,
             plot_contour_cbar=True,
             **kwargs):
    """
    Plot contour or contourf of the 2-d map, based on whether plot_contourf is set.

    Parameters
    ----------
    x_edges, y_edges: Arrays of edges, NOT the centers. The dim of edges should be len(z) + 1.
    z: 2-d binned map, or Column with data being the 2-d map.
    labels: None, str, or list of str. If None, no labels. If str, all labels are the same.
        If list, each level has a label except for the None values. This can be used to select levels to label,
        although using `manual` in `clabel_args` is more flexible.
    """
    if kwargs.get('z_cmap') is None:
        cmap = default_cmap
    else:
        cmap = kwargs['z_cmap']

    if plot_contourf:
        default_plt_args = dict(
            cmap=cmap,
        )
    else:
        default_plt_args = dict(
            colors='k',
            linewidths=0.8,
            linestyles='solid',
        )

    if plt_args is not None:
        default_plt_args.update(plt_args)
    plt_args = default_plt_args

    # cmap and colors shouldn't be set at the same time
    # set colors only when cmap is not set
    if plt_args.get('cmap') is not None and plt_args.get('colors') is not None:
        plt_args.pop('colors')

    levels = _calc_contour_levels(z, contour_levels, **kwargs)
    plt_args['levels'] = levels

    x_centers = (x_edges[1:] + x_edges[:-1]) / 2
    y_centers = (y_edges[1:] + y_edges[:-1]) / 2

    if plot_contourf:
        img = ax.contourf(x_centers, y_centers, z,
                          **plt_args,
                          )
    else:
        img = ax.contour(x_centers, y_centers, z,
                         **plt_args,
                         )

    if labels is not None:
        if not isinstance(labels, str) and len(labels) == len(img.levels) - 1:
            fmt = {}
            select_levels = []
            for level, string in zip(img.levels, labels):
                if string is not None:
                    select_levels.append(level)
                    fmt[level] = string
        else:
            fmt = labels
            select_levels = img.levels

        default_clabel_args = dict(
            inline=True,
            fontsize=10,
            fmt=fmt,
            levels=select_levels,
        )
        if clabel_args is not None:
            default_clabel_args.update(clabel_args)
        clabel_args = default_clabel_args

        ax.clabel(img, **clabel_args)

    breakpoint()
    if plot_contour_cbar:
        already_has_cbar = ax.figure.axes[-1].get_label() == '<colorbar>'
        already_has_cbar &= not plot_contourf
        already_has_cbar &= kwargs.get('cbar_ax', None) is None
        if already_has_cbar:
            cbar_ax = ax.figure.axes[-1]
            plt_colors = plt_args.get('colors')
            if isinstance(plt_colors, str):
                plt_colors = np.repeat(plt_colors, len(img.levels))
            lws = plt_args.get('linewidths')
            if isinstance(lws, (int, float)):
                lws = np.repeat(lws, len(img.levels))
            lss = plt_args.get('linestyles')
            if isinstance(lss, str):
                lss = np.repeat(lss, len(img.levels))
            for level, color, lw, ls in zip(img.levels, plt_colors, lws, lss):
                cbar_ax.axhline(level, color=color, linewidth=lw, linestyle=ls)
        else:
            cbar = plt.colorbar(img)
            cbar.set_label(kwargs['z_label'])

    return img


# %% func: plot scatter
@set_plot(special_suffix='scatter')
def scatter(ax,
            x, y, z=None, select=slice(None),
            plot_border=False,
            plt_args=None,
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
    )
    if plt_args is not None:
        default_plt_args.update(plt_args)
    plt_args = default_plt_args

    # plot the bg shadow first, then plot the colors
    if plot_border:
        border_args = plt_args.copy()
        border_args.pop('cmap', None)  # no cmap for border
        border_args.setdefault('edgecolors', 'k')
        border_args.setdefault('lw', border_args['s'] / 20)
        ax.scatter(x[select], y[select],
                   **border_args
                   )

    add_default_args = dict(lw=0)
    if z is not None:
        if kwargs.get('z_cmap') is not None:
            cmap = kwargs['z_cmap']
        else:
            cmap = default_cmap
        add_default_args.update(dict(
            c=z[select],
            vmin=kwargs['z_left'],
            vmax=kwargs['z_right'],
            cmap=cmap,
        ))
    add_default_args.update(plt_args)
    plt_args = add_default_args

    img = ax.scatter(x[select], y[select],
                     **plt_args
                     )
    ax.grid()

    return img


# %% func: hexbin
@set_plot(special_suffix='hexbin')
def hexbin(ax, x, y, z=None, select=slice(None),
           weights=None, func=calc.mean,
           bins=100,
           at_least=1, z_log=False,
           plt_args=None, **kwargs):
    """bins could be a number or a list of two numbers.
    """

    select = sel.combine(select, reference=x)
    x = x[select]
    y = y[select]
    if z is not None:
        z = z[select]
    if weights is not None:
        weights = weights[select]
        if z is None:
            z = weights
            if func is calc.mean:
                func = np.nansum
        else:
            # use index as fake z and get the real z in the function, along with weights
            from functools import partial
            func = partial(calc.value_in_bin, func=func, data=z, weights=weights)
            z = np.arange(len(z))

    if kwargs.get('z_cmap') is None:
        cmap = default_cmap
    else:
        cmap = kwargs['z_cmap']

    default_plt_args = dict(
        C=z,
        extent=(kwargs.get('x_left'), kwargs.get('x_right'), kwargs.get('y_left'), kwargs.get('y_right')),
        reduce_C_function=func,
        gridsize=bins, mincnt=at_least,
        cmap=cmap,
    )
    if z_log:
        default_plt_args['bins'] = 'log'

    if plt_args is not None:
        default_plt_args.update(plt_args)
    plt_args = default_plt_args

    img = ax.hexbin(x, y, **plt_args)
    return img


# %% func: bin_map
def bin_map(x, y, z=None, *,
            select=slice(None), title=None,
            plot_contour=1, plot_img=1,
            at_least=1, fill_nan=True,
            z_log=False,
            contour_levels=15,
            step_follow_window=False,
            contour_args=None, img_args=None,
            plot_contourf=False,
            savedir=default_savedir, filename=None,
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

    Currently there is no way to add contour lines to the cbar of imshow.

    Parameters
    ----------
    x, y, [z]: Column or simply 1-d array of data.
    plot_contour=1,  # 0: no contour, 1: contour of z, 2: contour of histogram
    plot_img=1,  # 0: no img, 1: img of z, 2: img of histogram
    contour_levels and plot_contourf is passed to plot._contour when plot_contour is set.
    fill_nan: if True, fill the nan values with at_least. Works only when z is None.

    In kwargs:
        everything of calc.bin_map, img and _contour

    Returns
    -------
    ax
    """
    if kwargs.get('plt_args') is not None:
        raise ValueError('plt_args is not allowed in bin_map. Use img_args and contour_args instead.')

    select = sel.combine(select, reference=x)
    z_map, x_edges, y_edges = calc.bin_map(x, y, z, select=select, at_least=at_least,
                                           step_follow_window=step_follow_window, **kwargs)
    if z_log:
        z_map = np.log10(z_map)
        at_least = np.log10(at_least)
        kwargs.setdefault('plot_cbar', False)
        kwargs.setdefault('plot_contour_cbar', False)

    if z is None and not step_follow_window:
        # calc histogram, but not the real number in each window
        kwargs.setdefault('plot_cbar', False)
        kwargs.setdefault('plot_contour_cbar', False)
    x_edges = attr.array2column(x_edges, meta_from=x)
    y_edges = attr.array2column(y_edges, meta_from=y)
    # edges is left here not converted into centers because this is done in scatter and img.

    if z is None:
        z_map = attr.array2column(z_map, name='hist')
        if kwargs.get('weights') is None:
            attr.set(z_map, label='counts')
        else:
            attr.set(z_map, label='weighted counts')

        if fill_nan:
            z_map_with_nan = attr.sift(z_map, min_=at_least, inplace=False)
        else:
            z_map_with_nan = z_map

        if plot_img:
            ax = img(x_edges, y_edges, z_map_with_nan,
                     filename='', plt_args=img_args, select=select, **kwargs)
            kwargs['ax'] = ax

        if plot_contour:
            z_map_with_nan = z_map_with_nan / calc.sum(z_map_with_nan)
            ax = _contour(
                x_edges, y_edges, z_map_with_nan, contour_levels=contour_levels,
                filename='', plt_args=contour_args, select=select, plot_contourf=plot_contourf, **kwargs)
    else:
        z_map = attr.array2column(z_map, meta_from=z)
        if plot_img:
            ax = img(x_edges, y_edges, z_map,
                     filename='', plt_args=img_args, select=select, **kwargs)
            kwargs['ax'] = ax
        if plot_contour == 1:
            ax = _contour(x_edges, y_edges, z_map, contour_levels=contour_levels, plot_contourf=plot_contourf,
                          filename='', plt_args=contour_args, select=select, **kwargs)
        elif plot_contour == 2:
            hist_map, x_edges, y_edges = calc.bin_map(x, y, **kwargs)
            ax = _contour(x_edges, y_edges, hist_map, contour_levels=contour_levels, plot_contourf=plot_contourf,
                          filename='', plt_args=contour_args, select=select, **kwargs)

    set_title_save_fig(ax=ax, x=x, y=y, z=z, title=title,
                       savedir=savedir, special_suffix='map', select=select, filename=filename)

    return ax


# %% func: contour_scatter
# to solve the problem of z_left and z_right in calculating the contour levels
@attr.set_values(
    set_default=True,
    to_set=['z_left', 'z_right'],
)
def contour_scatter(x, y, *,
                    select=slice(None), title=None,
                    contour_args=None, scatter_args=None,
                    contour_levels=15,
                    plot_scatter=True,
                    plot_contour=True,
                    plot_contourf=True,
                    savedir=default_savedir, filename=None,
                    **kwargs):
    """
    `plt.contour` is not good at dealing with sharp edges.
    `contour_levels` is passed to plot._contour.
    In kwargs:
        everything of calc.bin_map and scatter. `weights` is within kwargs and passed to calc.bin_map.
    Need to set filenames manually.
    """
    if kwargs.get('plt_args') is not None:
        raise ValueError('plt_args is not allowed in contour_scatter. Use scatter_args and contour_args instead.')

    ax = kwargs.pop('ax', plt.subplots()[1])

    z_map, x_edges, y_edges = calc.bin_map(x, y, select=select, **kwargs)
    z_map /= calc.sum(z_map)
    x_edges = attr.array2column(x_edges, meta_from=x)
    y_edges = attr.array2column(y_edges, meta_from=y)

    if plot_scatter:
        default_scatter_args = dict(
            s=10,
            rasterized=True,
            c='cornflowerblue',
            alpha=0.5,
            lw=0)
        if scatter_args is not None:
            default_scatter_args.update(scatter_args)
        scatter_args = default_scatter_args

        from contourpy import contour_generator
        from matplotlib import path
        cont_gen = contour_generator(z=z_map, x=(x_edges[1:] + x_edges[:-1]) / 2, y=(y_edges[1:] + y_edges[:-1]) / 2)
        levels_min = np.min(_calc_contour_levels(z_map, contour_levels, **kwargs))
        lines = cont_gen.lines(levels_min)
        p = path.Path(lines[0])
        select_outside = ~p.contains_points(np.column_stack([x, y]))

        if select != slice(None):
            select_outside &= sel.combine(select)

        ax = scatter(x, y, plt_args=scatter_args, select=select_outside, ax=ax, filename='', **kwargs)

    if plot_contour:
        ax = _contour(
            x_edges, y_edges, z_map,
            plt_args=contour_args, contour_levels=contour_levels, plot_contourf=plot_contourf,
            ax=ax, plot_cbar=False, plot_contour_cbar=False,
            filename='',
            **kwargs)

    set_title_save_fig(ax=ax, x=x, y=y, title=title,
                       savedir=savedir, special_suffix='cnt_sct', select=select, filename=filename)

    return ax


# %% func: plot 1d hist
@set_plot()
def _bar(ax, x, y, plt_args, y_log=False, **kwargs):
    if y_log:
        ax.set_yscale('log')
    plt_args.setdefault('width', x[1] - x[0])
    img = ax.bar(x, y, **plt_args)
    return img


def hist(x, *,
         select=slice(None), title=None,
         y_log=False,
         plot_errorbar=True,
         bar_args=None,
         errorbar_args=None,
         savedir=default_savedir, filename=None,
         **kwargs):
    """
    kwargs: everything in calc.hist
    """

    if kwargs.get('plt_args') is not None:
        raise ValueError('plt_args is not allowed in hist. Use errorbar_args and bar_args instead.')

    h, h_err, x_center = calc.hist(x, select=select, **kwargs)

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

    ax = _bar(x_center, h,
              plt_args=bar_args, y_log=y_log,
              filename='',
              **kwargs)

    if plot_errorbar:
        default_errorbar_args = dict(
            elinewidth=1.5,
            ecolor='k',
            markersize=0,
            ls='',
        )
        if errorbar_args is not None:
            default_errorbar_args.update(errorbar_args)
        errorbar_args = default_errorbar_args
        kwargs.pop('ax', None)
        ax = errorbar(x_center, h, y_err=h_err, ax=ax,
                      plt_args=errorbar_args,
                      filename='',
                      **kwargs)

    set_title_save_fig(ax=ax, x=x, title=title,
                       savedir=savedir, special_suffix='hist', select=select, filename=filename)

    return ax


# %% func: plot mean with std
@set_plot()
def errorbar(ax, x_centers, y_means, y_err=None,
             plt_args=None, **kwargs):
    """
    This is to plot mean with std, not for individual points.
    """

    default_plt_args = dict(
        marker='o', markersize=7,
        c='tab:blue',
    )
    if plt_args is not None:
        default_plt_args.update(plt_args)
    plt_args = default_plt_args

    img = ax.errorbar(x_centers, y_means, yerr=y_err, **plt_args)
    return img


def bin_x(x, y=None, *, y_log=False,
          mode='mean', bootstrap=0,
          select=slice(None), title=None,
          at_least=1,
          plot_scatter=False, plot_errorbar=True, plot_fill=True,
          errorbar_args=None, scatter_args=None, fill_args=None,
          savedir=default_savedir, filename=None,
          **kwargs):
    """
    If errorbar is not set, plot the filled area.
    kwargs: everything in calc.bin_x
    """

    if kwargs.get('plt_args') is not None:
        raise ValueError('plt_args is not allowed in bin_x. '
                         'Use errorbar_args,fill_args and scatter_args instead.')

    ys, y_err, x_centers = calc.bin_x(x, y, mode=mode, select=select, at_least=at_least, bootstrap=bootstrap, **kwargs)
    ys = attr.array2column(ys, meta_from=y)

    x_centers = attr.array2column(x_centers, meta_from=x)

    if errorbar_args is None:
        errorbar_args = dict()
    if plot_errorbar:
        y_err_used = y_err
    else:
        y_err_used = None
        errorbar_args.update({'markersize': 0})
    # `select` is still passed in order to show the title, but not used in the calculation
    ax = errorbar(x_centers, ys, y_err=y_err_used,
                  plt_args=errorbar_args, filename='', **kwargs)

    if not plot_errorbar and plot_fill:
        if fill_args is None:
            fill_args = dict()
        fill_args.setdefault('alpha', 0.2)
        if mode == 'median' or bootstrap == 1:
            ax.fill_between(x_centers, ys - y_err[0], ys + y_err[1], **fill_args)
        else:
            ax.fill_between(x_centers, ys - y_err, ys + y_err, **fill_args)
    kwargs.pop('ax', None)

    if y is not None and plot_scatter:
        if scatter_args is None:
            scatter_args = dict()
        scatter_args['c'] = 'silver'  # silver scatter in the background
        ax = scatter(x, y, z=None, ax=ax,
                     plt_args=scatter_args,
                     select=select,
                     filename='',
                     **kwargs)

    set_title_save_fig(ax=ax, x=x, y=y, title=title,
                       savedir=savedir, special_suffix='bin_x', select=select, filename=filename)

    return ax


# %% func: loess2d
@set_plot(special_suffix='loess')
def loess(ax, x, y, z, plt_args=None, plot_border=True,
          select=slice(None),
          **kwargs):
    select = sel.combine(select, reference=x)
    for i in [x, y, z]:
        select &= sel.good(i)
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
    )
    if plt_args is not None:
        default_plt_args.update(plt_args)
    plt_args = default_plt_args

    # plot the bg shadow first, then plot the colors
    default_border_args = dict(
        edgecolors='k',
        lw=4.5,
    )
    if plot_border:
        border_args = plt_args.copy()
        border_args.pop('cmap', None)
        default_border_args.update(border_args)
        border_args = default_border_args
        # border_args.setdefault('lw', border_args['s'] / 20)
        ax.scatter(x_use, y_use, **border_args)

    if kwargs.get('z_cmap') is not None:
        cmap = kwargs['z_cmap']
    else:
        cmap = default_cmap

    add_default_args = dict(lw=0)
    add_default_args.update(dict(
        c=zout,
        cmap=cmap,
    ))
    add_default_args.update(plt_args)
    plt_args = add_default_args

    img = ax.scatter(x_use, y_use, **plt_args)
    return img
