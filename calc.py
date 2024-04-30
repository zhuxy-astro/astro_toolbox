#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# %% import
import sys

import numpy as np
from astropy.cosmology import Planck15, FlatLambdaCDM
from scipy.interpolate import interp1d
from scipy.spatial import KDTree
from scipy.optimize import curve_fit as scipy_curve_fit
from scipy import stats

from . import attr, misc


# %% func: select good values
def select_good(array):
    # returns an array of bools indicating whether the values are not masked, not nan and not infinite.
    if hasattr(array, 'copy'):
        # I don't know why but this function will change the original array without copying.
        array = array.copy()

    if hasattr(array, 'mask'):
        mask = array.mask
    else:
        mask = None
    if mask is None:
        mask = np.zeros_like(array, dtype=bool)
    mask |= ~np.isfinite(array)
    return ~mask


def good_data_weights(data=None, weights=None):
    if data is None:
        assert weights is not None, "Both data and weights are None!"
        np_data = np.ones_like(weights)
        good_ind = np.ones_like(weights, dtype=bool)
    else:
        good_ind = select_good(data)
        np_data = np.array(data)  # avoid changing the original array
        if weights is None:
            return np_data[good_ind], None

    np_weights = np.array(weights)  # avoid changing the original array
    np_data, np_weights = np_data.flatten(), np_weights.flatten()
    assert np.shape(np_data) == np.shape(np_weights), \
        'The shape of data and weights does not match!'

    good_ind = good_ind & select_good(np_weights)
    return np_data[good_ind], np_weights[good_ind]


def weights_is_none(weights):
    if weights is None:
        return True
    elif np.nansum(weights) == 0:
        return True
    elif not any(select_good(weights)):
        return True
    return False


def good_values(array):
    good_ind = select_good(array)
    np_array = np.array(array)  # avoid changing the original array
    return np_array[good_ind]


# %% func: min, max, mean, sum, std
def min(data):
    return np.nanmin(good_values(data))


def max(data):
    return np.nanmax(good_values(data))


def sum(data, weights=None):
    np_data, np_weights = good_data_weights(data, weights)
    if weights_is_none(np_weights):
        return np.nansum(np_data)
    return np.nansum(np_data * np_weights)


def mean(data, weights=None):
    np_data, np_weights = good_data_weights(data, weights)
    if weights_is_none(np_weights):
        return np.average(np_data)
    return np.average(np_data, weights=np_weights)


def std(data, weights=None):
    """ddof=1
    """
    np_data, np_weights = good_data_weights(data, weights)
    if weights_is_none(np_weights):
        return np.std(np_data, ddof=1)
    average = mean(np_data, weights=np_weights)
    variance = mean((np_data - average)**2, weights=np_weights)
    N = len(np_data)
    return np.sqrt(variance * N / (N - 1))


# %% func: vmax_inv
Planck_H0 = Planck15.H0
Planck_Om0 = Planck15.Om0


def vmax_inv(z_left, z_right, z_min, z_max,
             z=None,
             fill_nan=False,
             max_vmax_inv=100,
             H0=Planck_H0, Om0=Planck_Om0
             ):
    """
    z_left and z_right are the redshift range limit we want to consider. Float.
    z_min  and z_max   are data inferred from the observation. Could be float or np.ndarray.
    When z is set, there is another selection of z_left< z < z_right. The others are set to nan.
    fill_nan is to fill the invalid z_min and z_max with weight=1.
    The weight is always returned as an array, 0-dimensional when the input are floats.
    """
    cosmo = FlatLambdaCDM(H0=H0, Om0=Om0)
    # use interpolating in order to avoid calculating the integral many times
    zs = np.linspace(0., 1., 1000)
    zs = zs ** 2
    zs = zs * 10.
    ds = cosmo.comoving_distance(zs).value
    dist = interp1d(zs, ds)

    def _V(z1, z2):
        return dist(z2) ** 3 - dist(z1) ** 3
    V0 = _V(z_left, z_right)
    Vmax = _V(np.maximum(z_left, z_min), np.minimum(z_right, z_max))
    if fill_nan:
        # nan values of z_min and z_max
        select_bad_zmin_zmax = (~select_good(z_min)) | (~select_good(z_max))
        if z is not None:
            # wrong values of z_min and z_max
            select_bad_zmin_zmax |= (z > z_max) | (z < z_min)
        Vmax = np.where(select_bad_zmin_zmax, V0, Vmax)
    if z is not None:
        Vmax = np.where((z < z_left) | (z > z_right), np.nan, Vmax)
    # give nan when Vmax is negative or zero, encountered when the galaxies' z range outside the considered z range.
    Vmax = np.where(Vmax < 1e-5, np.nan, Vmax)
    weight = V0 / Vmax
    weight = np.where((weight > max_vmax_inv) & ~np.isnan(weight),
                      max_vmax_inv, weight)
    return weight


# %% func: ba_to_incl
def ba_to_incl(ba, alpha=0.15):
    return np.arcsin(np.sqrt((1 - ba ** 2) / (1 - alpha ** 2))) * 180 / np.pi


# %% func: deg_to_radian
def deg_to_radian(deg, reset_zero=False):
    if reset_zero:
        deg = deg - 360 * (deg > 180)
    return deg / 180 * np.pi


# %% func: double_linear
def double_linear(xs, m, n, c, yc, amp):
    """double_linear

    Parameters
    ----------
    x : ndarray
        the input xs
    m : float
        slope for x < c
    n : float
        slope for x >= c
    c : float
        turn-over point
    yc : float
        y value at x = c
    amp : float
        global amplitude

    Returns
    -------
    the y values corresponding to the input xs
    shape (len(xs))
    """
    ys = np.empty(len(xs))
    ys[xs < c] = (yc + m * xs - m * c)[xs < c]
    ys[xs >= c] = (yc + n * xs - n * c)[xs >= c]
    return amp * ys


# %% func: schechter
def schechter(xs, amp, x_c, alpha):
    """schechter

    Parameters
    ----------
    xs : ndarray
        the input xs
    amp : float
        the amplitude
    x_c : float
        turn-over point
    alpha : float
        the slope of the power law part

    Returns
    -------
    the y values corresponding to the input xs
    shape (len(xs))
    """
    return amp * (xs / x_c)**alpha * np.exp(-xs / x_c) / x_c


def schechter_log(logx, amp, logx_c, alpha):
    """schechter_log

    Parameters
    ----------
    logx : ndarray
        the input log(x)
    amp : float
        the amplitude
    logx_c : float
        the log of the turn over point
    alpha : float
        the slope of the power law part

    Returns
    -------
    the y values corresponding to the input xs
    shape (len(logx))
    """
    return (np.log(10) * amp
            * np.exp(-10**(logx - logx_c))
            * (10**((alpha + 1) * (logx - logx_c))))


# %% func: curve_fit
def curve_fit(f, xdata, ydata, *args, **kwargs):
    select_good_xy = select_good(xdata) & select_good(ydata)
    return scipy_curve_fit(f, xdata[select_good_xy], ydata[select_good_xy], *args, **kwargs)


# %% func: fraction
def fraction(select, within=slice(None), weights=None, print_info=False):
    """In simple calculating the fraction without `within` and just `weights`, the function is equivalent to `mean`.
    """
    # combine `within` when it is a list, and save `within` from slice(None)
    within = attr.combine_selections(within, reference=select)
    within = np.array(within, dtype=bool)
    select = np.array(select, dtype=bool)
    within = within & select_good(select)

    if weights is None:
        denominator = np.nansum(within)
        numerator = np.nansum(select & within)
    else:
        denominator = np.nansum(weights[within])
        numerator = np.nansum(weights[select & within])

    if print_info:
        print(f'{numerator} / {denominator}')
    return numerator / denominator


# %% func: select_value_edges
def select_value_edges(data, edges, name=None, math=False):
    """return the list of select arrays cut by the edges, with length = len(edges) + 1
    """
    if name is None:
        try:
            name = data.name
        except AttributeError:
            name = 'x'
    if math:
        le = r'\leq'
    else:
        le = '<='
    select_list = []
    select_list.append(attr.array2column(
        data <= edges[0], name=f'{name}{le}{edges[0]:.3g}'))
    for i in range(len(edges) - 1):
        select_name = f'{edges[i]:.3g}<{name}{le}{edges[i + 1]:.3g}'
        if math:
            select_name = r'$' + select_name + r'$'
        select_list.append(attr.array2column(
            (data > edges[i]) & (data <= edges[i + 1]),
            name=select_name))
    name = f'{name}>{edges[-1]:.3g}'
    if math:
        select_name = r'$' + name + r'$'
    select_list.append(attr.array2column(
        data > edges[-1], name=select_name))
    return select_list


# %% func: weighted_percentile and median
def weighted_percentile(data=None, weights=None,
                        select=slice(None), percentile=0.5):
    """in default calculate the weighted median.

    `percentile` is the percentile of the data (or weights if data=None) ranking from small to large.
    It can be either a float or a list of floats no larger than 1.
    The returned value will be consistent with percentile, i.e., a list or a float.

    When data=None, weights is treated as a histogram, and the threshold is calculated such that
    the sum of the normalized hist<threshold meets the percentile. For binned scatters, it marks the
    threshold OUT of which the number of points meets the percentile.
    When data is set, calculate the weighted percentile directly.

    Some contents refer to
    https://github.com/tinybike/weightedstats/blob/master/weightedstats/__init__.py
    """
    # when calculating the percentile in a histogram, only the weights are given.
    data_is_none = data is None
    if data_is_none:
        data = np.arange(np.array(weights).size)
    else:
        data = data.copy()

    # select values
    select = attr.combine_selections(select, reference=data)
    data = np.where(select, data, np.nan)

    # when weights is not set, calculate the percentiles directly
    if weights is None:
        return np.nanpercentile(good_values(data), np.array(percentile) * 100.)
    else:
        weights = weights.copy()

    data, weights = good_data_weights(data, weights)

    # when there is no available weights, return nan directly to save time.
    if not any(weights > 0):
        if isinstance(percentile, (float, int)):
            return np.nan
        else:
            return np.full(len(percentile), np.nan)

    if data_is_none:
        sorted_ind = weights.argsort()
        sorted_weights = weights[sorted_ind]
        returned_data = sorted_weights
    else:
        sorted_ind = data.argsort()
        sorted_weights = weights[sorted_ind]
        returned_data = data[sorted_ind]

    # Calculate the cumulative sum of the weights
    cum_weights = np.cumsum(sorted_weights)
    normalize = cum_weights[-1]
    cum_weights = cum_weights / normalize

    # if calculating the median, find possible large weight to save time
    if isinstance(percentile, float) and percentile == 0.5:
        exist_a_large_weight = any(weights > 0.5 * normalize)
        if exist_a_large_weight:
            return (data[weights == np.max(weights)])[0]

    indices = np.searchsorted(cum_weights, percentile)
    if isinstance(percentile, float) and percentile == 0.5:
        if np.abs(cum_weights[indices - 1] - 0.5) < sys.float_info.epsilon:
            return np.mean(returned_data[indices - 1:indices + 1])
    return returned_data[indices]


def median(data, weights=None):
    return weighted_percentile(data, weights=weights)


# %% func: select percentile
def select_percentile(x,
                      percentile=[0.25, 0.50, 0.75],
                      weights=None,
                      select=slice(None)):
    """percentile must be a list or array
    """
    percentile = np.array(percentile)
    cuts = weighted_percentile(data=x, weights=weights,
                               select=select,
                               percentile=percentile)
    nbins = len(cuts) + 1
    select_percentile = np.ones((nbins, len(x)), dtype=bool)
    for i in range(nbins):
        if i != 0:
            select_percentile[i] &= (x > cuts[i - 1])
        if i != nbins - 1:
            select_percentile[i] &= (x < cuts[i])
    return select_percentile


# %% func: binning
@attr.set_values(
    set_default=True,
    to_set=['x_left', 'x_right', 'x_step']
)
def binning(x, bins=50, x_step=None,
            x_left=None, x_right=None,
            select_index=True):
    """
    By default use bins=50, and set x_step using the range of x.
    But as long as x_step is set, even within x.meta, bins will be overwritten.
    x_left=min(x), x_right=max(x)
    select_index: whether to return the index in each bin, used in functions only.

    Returns
    -------
    bin_edges, select_index_in_bin
    select_index_in_bin = None if select_index is False
    """
    if x_step is None:
        # use bins
        bin_edges = np.linspace(x_left, x_right, bins + 1)
        # x_step = bin_edges[1] - bin_edges[0]
    else:
        bin_edges = np.arange(x_left, x_right + x_step, x_step)
        bins = len(bin_edges) - 1

    if not select_index:
        return bin_edges, None

    digitized_x = np.digitize(x, bin_edges)
    select_index_in_bin = [(digitized_x == i) for i in range(1, bins + 1)]
    # the following line should be a little faster using numpy matrix
    # select_index_in_bin = np.eye(bins + 1)[digitized_x - 1].astype(bool)

    return bin_edges, select_index_in_bin


# %% func: value_in_bin
def value_in_bin(index_in_bin=None, data=None, weights=None,
                 at_least=1, func=mean, bar=lambda: None, bootstrap=0):
    """bootstrap: 0: no bootstrap; 1: mean of the bootstrap;
        2: std of the bootstrap; 3: confidence interval; 4: return the res object.
    """
    if index_in_bin is None:
        index_in_bin = slice(None)
    if data is None:
        data = np.ones_like(index_in_bin)
    data_in_bin = data[index_in_bin]  # usually used in calculating hist when only weights are given

    select = select_good(data_in_bin)
    if weights is not None:
        weights_in_bin = weights[index_in_bin]
        select = select & select_good(weights_in_bin)
        select = select & (weights_in_bin != 0.)

    bootstrap_args = dict(
        n_resamples=500,
        # vectorized=False,
        paired=True,
        confidence_level=0.68,
        random_state=42
    )

    def bar_and_return(value, bar=bar):
        bar()
        return value

    if bootstrap and (at_least <= 1):
        at_least = 2

    if select.sum() < at_least:
        if bootstrap == 1:
            return bar_and_return(np.nan)
        elif bootstrap == 2:
            return bar_and_return(np.nan)
        elif bootstrap == 3:
            return bar_and_return(np.array([np.nan, np.nan]))
        elif bootstrap == 4:
            return bar_and_return(stats.bootstrap(([np.nan, np.nan],), func, **bootstrap_args))

        # no bootstrap
        if weights is None:
            func_return_test = func(np.array([1., 1.]))
        else:
            func_return_test = func(np.array([1., 1.]), weights=np.array([1., 1.]))
        return_is_single = not hasattr(func_return_test, '__len__')
        if return_is_single:
            return bar_and_return(np.nan)
        else:
            return bar_and_return(np.full(len(func_return_test), np.nan))

    if not bootstrap:
        if weights is None:
            return bar_and_return(func(data_in_bin))
        else:
            return bar_and_return(func(data_in_bin, weights=weights_in_bin))

    # bootstrap
    if weights is None:
        bootstrap_data = (data_in_bin,)
    else:
        bootstrap_data = (data_in_bin, weights_in_bin)

    try:
        res = stats.bootstrap(bootstrap_data, func, **bootstrap_args)
    except Exception:
        bootstrap_args['paired'] = False
        res = stats.bootstrap(bootstrap_data, func, **bootstrap_args)

    if bootstrap == 1:
        return bar_and_return(np.nanmean(res.bootstrap_distribution))
    elif bootstrap == 2:
        return bar_and_return(res.standard_error)
    elif bootstrap == 3:
        return bar_and_return(res.confidence_interval)
    elif bootstrap == 4:
        return bar_and_return(res)


# %% func: bin_x
@attr.set_values(
    set_default=True,
    to_set=['x_left', 'x_right', 'x_step']
)
def bin_x(x, y, weights=None,
          mode='mean',
          median_percentile=[0.841, 0.159],
          func=None, errfunc=None,
          bins=20, x_step=None,
          x_left=None, x_right=None,
          select=slice(None),
          bootstrap=False,
          at_least=1,
          **kwargs):
    """
    `mode` could be 'mean', 'meanerr' or 'median'. Overwritten by `func` and `errfunc`.
    When mode is 'median', median_percentile is used to calculate the errors.
    `func` and `errfunc` should be functions that take 1-d array (may with a `weights` array) and return a number.
    Returns
    -------
    ys, y_err, x_centers
    """

    select = attr.combine_selections(select, reference=x)
    x = x[select]
    y = y[select]
    weights = weights[select] if weights is not None else None

    x_edges, index_in_bin = binning(
        x, x_left=x_left, x_right=x_right, bins=bins, x_step=x_step)

    if func is None:
        if mode == 'mean' or mode == 'meanerr':
            func = mean
        elif mode == 'median':
            func = median
    if errfunc is None:
        if mode == 'mean':
            errfunc = std
        elif mode == 'meanerr':
            ...
            errfunc = std  # TODO
        elif mode == 'median':
            from functools import partial
            errfunc = partial(weighted_percentile, percentile=np.sort(median_percentile))

    value_in_bin_kwargs = dict(func=func, at_least=at_least)
    if weights is not None:
        value_in_bin_kwargs['weights'] = weights

    if bootstrap:
        with misc.Bar(len(x_edges) - 1) as bar:
            value_in_bin_kwargs['bootstrap'] = 4
            ress = [value_in_bin(ind, y, bar=bar, **value_in_bin_kwargs) for ind in index_in_bin]

        ys = [np.nanmean(res.bootstrap_distribution) for res in ress]
        y_err = [res.confidence_interval for res in ress]

    else:
        value_in_bin_kwargs['bootstrap'] = 0
        ys = [value_in_bin(ind, y, **value_in_bin_kwargs) for ind in index_in_bin]
        value_in_bin_kwargs['func'] = errfunc
        y_err = [value_in_bin(ind, y, **value_in_bin_kwargs) for ind in index_in_bin]

    if mode == 'median' or bootstrap:
        y_err = np.array(y_err).T
        y_err = np.abs(y_err - ys)

    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    return ys, y_err, x_centers


# %% func: bin_map
@attr.set_values(
    set_default=True,
    to_set=[
        'x_left', 'x_right', 'x_step', 'x_window',
        'y_left', 'y_right', 'y_step', 'y_window'
    ]
)
def bin_map(x, y, z=None, weights=None, func=mean,
            at_least=1, select=slice(None),
            step_follow_window=False,
            x_left=None, x_right=None,
            x_step=None, x_window=None,
            y_left=None, y_right=None,
            y_step=None, y_window=None,
            **kwargs  # in order to protect from other params
            ):
    """
    Binning x and y to calculate some function of z.
    If z is not set, return the 2-d histogram.

    Parameters
    ----------
    x, y, [z, weights]: columns, or simply 1-d array of data.

    func: in each bin, the function used to calculate the value. Must satisfy func(data, weights) or func(data).

    select : select the data.

    at_least : at least how many selected samples in each window? The windows under this value is filled with np.nan.

    step_follow_window : if True, the step will be overwritten by the window.

    x_left=min(x), x_right=max(x),
    y_left=min(y), y_right=max(y),
    x_step=(x_right - x_left) / 40)
    y_step=(y_right - y_left) / 40)
    x_window=x_step, y_window=y_step,

    Returns
    -------
    binned_map: an np.array of the 2-d map,
                left and right from the min and max of the map,
                and other attributes inherited from z.
    x_bin_edges, y_bin_edges: 1-d array with length = nbins+1. Similar with np.histogram2d
    """

    select = attr.combine_selections(select, reference=x)
    x = x[select]
    y = y[select]
    if z is not None:
        z = z[select]
        assert (len(x) == len(z)), "The length of input array doesn't match!"

    if weights is not None:
        weights = weights[select]

    assert (len(x) == len(y)), "The length of input array doesn't match!"
    data_length = len(x)

    default_step_num = 40
    if x_step is None:
        x_step = (x_right - x_left) / default_step_num
    if x_window is None:
        x_window = x_step
    if y_step is None:
        y_step = (y_right - y_left) / default_step_num
    if y_window is None:
        y_window = y_step

    if step_follow_window:
        x_step = x_window
        y_step = y_window

    x_bin_edges = np.arange(x_left, x_right + x_step, x_step)
    y_bin_edges = np.arange(y_left, y_right + y_step, y_step)

    if z is None and weights is None and step_follow_window:
        # calculate the real, unweighted histogram here
        binned_map, x_bin_edges, y_bin_edges = np.histogram2d(
            x, y, bins=(x_bin_edges, y_bin_edges),
            range=[[x_left, x_right], [y_left, y_right]]
        )
        binned_map = binned_map.T
        return binned_map, x_bin_edges, y_bin_edges

    if z is None:
        # calc hist
        at_least = 0  # avoid nan in bin_map if calc hist
        z = np.ones_like(x)
        func = sum

    z = np.array(z)

    xbins = len(x_bin_edges) - 1
    ybins = len(y_bin_edges) - 1
    x_window_left = x_bin_edges[:-1] + (x_step - x_window) / 2.
    y_window_left = y_bin_edges[:-1] + (y_step - y_window) / 2.
    x_window_right = x_window_left + x_window
    y_window_right = y_window_left + y_window

    index_in_x_bin = np.zeros([xbins, data_length], dtype=bool)
    index_in_y_bin = np.zeros([ybins, data_length], dtype=bool)
    for i in range(xbins):
        index_in_x_bin[i] = x > x_window_left[i]
        index_in_x_bin[i] &= x <= x_window_right[i]
    for j in range(ybins):
        index_in_y_bin[j] = y > y_window_left[j]
        index_in_y_bin[j] &= y <= y_window_right[j]

    binned_map = np.zeros([xbins, ybins])

    """
    Using multiprocessing here will cause bugs like `TypeError: cannot pickle '_io.TextIOWrapper' object`.
    Using other multi-processing tools may help, but I haven't tried much of them.
    `multiprocess` works but is not really faster. I have no idea why.

    Using numpy matrix here is possible, but need the broadcasting of z and
    weights to 3d arrays of shape (len(z), xbins, ybins), which is not really
    faster, uses more memory, and does not support more complicated functions.
    """
    with misc.Bar(xbins * ybins) as bar:
        for i in range(xbins):
            for j in range(ybins):
                index_in_xy_bin = index_in_x_bin[i] & index_in_y_bin[j]
                index_in_xy_bin = np.where(index_in_xy_bin)[0]
                """
                The above line takes some time, but eliminates the calculation of
                z[index_in_xy_bin] and weights[index_in_xy_bin] in `value_in_bin`,
                which are even slower when the selection is sparse.
                """
                binned_map[i][j] = value_in_bin(index_in_xy_bin, data=z,
                                                weights=weights, at_least=at_least, func=func, bar=bar)

    # transpose the matric to follow a Cartesian convention. The same operation is NOT done in, e.g., np.histogram2d.
    binned_map = binned_map.T
    return binned_map, x_bin_edges, y_bin_edges


# %% func: hist
@attr.set_values(
    set_default=True,
    to_set=['x_left', 'x_right', 'x_step']
)
def hist(x, weights=None,
         bins=100, x_step=None,
         x_left=None, x_right=None,
         at_least=0,
         density=False, norm=False, select=slice(None), **kwargs):
    """
    density: scaled by the step length
    norm: further scaling to make integrated area = 1. Overwrites density.
    bins=100
    x_step=None, overwrite bins when set
    x_left=min(x), x_right=max(x)

    Returns
    -------
    hist, hist_err, bin_center
    """
    select = attr.combine_selections(select, reference=x)
    x = x[select]

    bin_edges, index_in_bin = binning(
        x, bins=bins, x_step=x_step, x_left=x_left, x_right=x_right, select_index=True)

    if weights is None:
        hist, bin_edges = np.histogram(
            x, bins=bin_edges,
            range=(x_left,
                   x_right),
            density=False  # density here is manually corrected in the following
        )
        hist_err = np.sqrt(hist)
    else:
        weights = weights[select]

        hist = [value_in_bin(index_in_bin_i, weights=weights, at_least=at_least, func=sum)
                for index_in_bin_i in index_in_bin]
        hist_err = [value_in_bin(index_in_bin_i, weights=weights, at_least=at_least,
                                 func=lambda d, weights: np.sqrt(sum(weights ** 2)))
                    for index_in_bin_i in index_in_bin]

    hist = np.array(hist)
    hist_err = np.array(hist_err)
    scale = 1.
    if density:
        scale = 1. / x_step
    if norm:
        scale = 1. / np.nansum(hist)
    hist = hist * scale
    hist_err = hist_err * scale

    bin_center = (bin_edges[:-1] + bin_edges[1:]) / 2

    return hist, hist_err, bin_center


# %% loess2d from Kai
# forked from https://github.com/wkcosmology/MyPyScript/blob/master/stats/loess2d.py
"""
usage:
    loess = LOESS2D(x, y, z, 30)
    z_smoothed = loess(x, y)
"""


class LOESS2D:
    def __init__(self, x, y, val, n_nbr, frac=0.5, boxsize=None, xy_ratio=1):
        """
        x, y, val: ndarray with shape (N, )
            input coordinate and values
        n_nbr: number of neighbours for smoothing, or fraction w.r.t. to the total population
            # of neighbors = (n_nbr >= 1) ? int(n_nbr) : int(n_nbr * N)
        boxsize: optional
            if assigned a value, the distance is calculated in a periodic box
        xy_ratio:
            weight in the calculation of distance
            d = sqrt(xy_ratio * (x_1 - x_2)^2 + (y_1 - y_2)^2)^2
        """
        # Record the transformation for x and y coordinates
        self._xnorm = self._gen_norm(x, xy_ratio)
        self._ynorm = self._gen_norm(y)
        self._xn = self._xnorm(x)
        self._yn = self._ynorm(y)
        self._val = val.copy()
        self._tree = KDTree(
            np.column_stack((self._xn, self._yn)), copy_data=True, boxsize=boxsize
        )
        if n_nbr >= 1:
            self._n_nbr = int(n_nbr)
        else:
            self._n_nbr = int(frac * len(x))
        if self._n_nbr > len(x):
            raise Exception(
                "Number of smoothing neighbors exceeds the total number of points"
            )
        print("# of neightbours for smoothing: %d" % self._n_nbr)

    def __call__(self, x, y):
        x_norm = self._xnorm(x)
        y_norm = self._ynorm(y)
        d_nbr, i_nbr = self._tree.query(np.column_stack((x_norm, y_norm)), self._n_nbr)
        d_norm = (d_nbr.T / d_nbr[:, -1]).T
        weight = (1 - d_norm**3) ** 3
        val = np.sum(weight * self._val[i_nbr], axis=1) / np.sum(weight, axis=1)
        return val

    def _gen_norm(self, arr, ratio=1):
        """
        Normalize the coordinate using quantiles rather than the standard deviation
        to avoid the impact of outliners.
        """
        xl, x_med, xu = np.quantile(arr, [0.17, 0.5, 0.84])
        return lambda x: (arr - x_med) / (xu - xl) * ratio
