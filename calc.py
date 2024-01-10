#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# %% import
import sys
import numpy as np
from astropy.cosmology import Planck18, FlatLambdaCDM
from scipy.interpolate import interp1d
# from alive_progress import alive_bar
from astropy.utils.console import ProgressBar
from . import attr
from scipy.spatial import KDTree
from scipy.optimize import curve_fit as scipy_curve_fit


# %% func: choose good and calc min, max, mean and median
def isgood(array):
    # returns the indices of values that are not masked, not nan and not infinite.
    try:
        # I don't know why but this function will change the original array.
        array = array.copy()
    except Exception:
        pass

    try:
        mask = array.mask
    except Exception:
        mask = None
    if mask is None:
        mask = np.zeros_like(array, dtype=bool)
    mask |= ~np.isfinite(array)
    return ~mask


def select_good_values(array):
    good_ind = isgood(array)
    np_array = np.array(array)  # avoid changing the original array
    return np_array[good_ind]


def min(array):
    return np.nanmin(select_good_values(array))


def max(array):
    return np.nanmax(select_good_values(array))


def mean(array):
    return np.nanmean(select_good_values(array))


def median(array):
    return np.nanmedian(select_good_values(array))


# %% func: vmax_inv
Planck_H0 = Planck18.H0
Planck_Om0 = Planck18.Om0


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
    # this can make V_max in the relative accuracy of ~1e-4
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
        select_bad_zmin_zmax = (~isgood(z_min)) | (~isgood(z_max))
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


# %% func: to_radian
def to_radian(deg, reset_zero=False):
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
    xdata = np.array(xdata.copy())
    ydata = np.array(ydata.copy())
    select_good = np.isfinite(xdata) & np.isfinite(ydata)
    return scipy_curve_fit(f, xdata[select_good], ydata[select_good], *args, **kwargs)


# %% func: bpt
"""
OIII range: -1.2 ~ 1.5
NII range: -2 ~ 1
SII range: -1.2 ~ 0.8
OI range: -2.2 ~ 0
"""


def bpt_nii_agn_comp(nii_ha):
    return 0.61 / (nii_ha - 0.47) + 1.19


def bpt_nii_sf_comp(nii_ha):
    return 0.61 / (nii_ha - 0.05) + 1.3


def bpt_nii(nii_ha, oiii_hb):
    """
    return: array of ints. 0: sf, 1: composite, 2: agn, nan: nan.
    """
    result = np.empty_like(nii_ha)
    result[:] = np.nan

    select_sf = oiii_hb < bpt_nii_sf_comp(nii_ha)
    result[select_sf] = 0
    select_agn = oiii_hb > bpt_nii_agn_comp(nii_ha)
    result[select_agn] = 2
    select_comp = oiii_hb >= bpt_nii_sf_comp(nii_ha)
    select_comp &= oiii_hb <= bpt_nii_agn_comp(nii_ha)
    result[select_comp] = 1
    return result


def bpt_sii_sf_agn(sii_ha):
    return 0.72 / (sii_ha - 0.32) + 1.30


def bpt_sii_liner_seyfert(sii_ha):
    return 1.89 * sii_ha + 0.76


def bpt_sii(sii_ha, oiii_hb):
    """
    return: array of ints. 0: sf, 1: liner, 2: seyfert, nan: nan.
    """
    result = np.empty_like(sii_ha)
    result[:] = np.nan

    select_sf = oiii_hb < bpt_sii_sf_agn(sii_ha)
    result[select_sf] = 0
    select_seyfert = oiii_hb >= bpt_sii_sf_agn(sii_ha)
    select_seyfert &= oiii_hb >= bpt_sii_liner_seyfert(sii_ha)
    result[select_seyfert] = 2
    select_liner = oiii_hb >= bpt_sii_sf_agn(sii_ha)
    select_liner &= oiii_hb < bpt_sii_liner_seyfert(sii_ha)
    result[select_liner] = 1
    return result


def bpt_oi_sf_agn(oi_ha):
    return 0.73 / (oi_ha + 0.59) + 1.33


def bpt_oi_liner_seyfert(oi_ha):
    return 1.18 * oi_ha + 1.30


def bpt_oi(oi_ha, oiii_hb):
    """
    return: array of ints. 0: sf, 1: liner, 2: seyfert, nan: nan.
    """
    result = np.empty_like(oi_ha)
    result[:] = np.nan

    select_sf = oiii_hb < bpt_oi_sf_agn(oi_ha)
    result[select_sf] = 0
    select_seyfert = oiii_hb >= bpt_oi_sf_agn(oi_ha)
    select_seyfert &= oiii_hb >= bpt_oi_liner_seyfert(oi_ha)
    result[select_seyfert] = 2
    select_liner = oiii_hb >= bpt_oi_sf_agn(oi_ha)
    select_liner &= oiii_hb < bpt_oi_liner_seyfert(oi_ha)
    result[select_liner] = 1
    return result


# %% func: weighted_median
def weighted_median(data, weights=None):
    # some contents refer to
    # https://github.com/tinybike/weightedstats/blob/master/weightedstats/__init__.py
    if weights is None:
        return median(data)
    data, weights = np.array(data).flatten(), np.array(weights).flatten()
    assert np.shape(data) == np.shape(weights), \
        'The shape of data and weights does not match!'

    # remove nan
    use_index = ~ (np.isnan(data) | np.isnan(weights))
    data, weights = data[use_index], weights[use_index]

    if not any(weights > 0):
        return np.nan

    sorted_ind = data.argsort()
    sorted_data = data[sorted_ind]
    sorted_weights = weights[sorted_ind]
    # Calculate the cumulative sum of the weights
    cum_weights = np.cumsum(sorted_weights)
    # Find the median weight
    midpoint = cum_weights[-1] / 2

    exist_a_large_weight = any(weights > midpoint)
    if exist_a_large_weight:
        return (data[weights == np.max(weights)])[0]

    median_ind = np.searchsorted(cum_weights, midpoint)
    if np.abs(cum_weights[median_ind - 1] - midpoint) < sys.float_info.epsilon:
        return np.mean(sorted_data[median_ind - 1:median_ind + 1])
    return sorted_data[median_ind]

    # manual alternative of np.searchsorted:
    # for i in range(len(cum_weights)):
    #     if cum_weights[i] >= midpoint:
    #         return sorted_data[i]


# %% func: select percentile
def select_percentile(x,
                      array_of_percentiles=[25, 50, 75],
                      select=slice(None)):
    # x = choose_value(dict(), None, x, 'data', None, x)
    x = np.where(select, x, np.nan)
    cuts = np.nanpercentile(x, array_of_percentiles)
    nbins = len(cuts) + 1
    select_percentile = np.ones((nbins, len(x)), dtype=bool)
    for i in range(nbins):
        if i != 0:
            select_percentile[i] &= (x > cuts[i - 1])
        if i != nbins - 1:
            select_percentile[i] &= (x < cuts[i])
    return select_percentile


# %% func: bin map
def value_in_bin(data, index_in_bin, weights=None,
                 at_least=1, func=mean, bar=None):
    data_in_bin = data[index_in_bin]
    if np.isfinite(data_in_bin).sum() < at_least:
        value = np.nan
        if bar is not None:
            bar.update()
        # bar()
        # continue
        return value
    if weights is not None:
        weights_in_bin = weights[index_in_bin]
        value = func(data_in_bin, weights_in_bin)
    else:
        value = func(data_in_bin)
    if bar is not None:
        bar.update()
    # bar()
    return value


@attr.set_values(to_set=[
    'x_left', 'x_right', 'x_step', 'x_window',
    'y_left', 'y_right', 'y_step', 'y_window'
])
def bin_map(x, y, z=None, weights=None, func=mean,
            at_least=1, select=slice(None),
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

    func: in each bin, the function used to calculate the value.

    select : select the data.

    at_least : at least how many selected samples in each window? The windows under this value is filled with np.nan.

    other kwargs:
        x_left=min(x), x_right=max(x),
        y_left=min(y), y_right=max(y),
        x_step=(x_right - x_left) / 20)
        y_step=(y_right - y_left) / 20)
        x_window=x_step, y_window=y_step,

    Returns
    -------
    binned_map: an np.array of the 2-d map,
                left and right from the min and max of the map,
                and other attributes inherited from z.
    x_edges, y_edges: 1-d array with length = nbins+1. Similar with np.histogram2d
    """

    x = x[select]
    y = y[select]
    if z is not None:
        z = z[select]
        assert (len(x) == len(z)), "The length of input array doesn't match!"

    # weights_data = choose_value(kwargs, None, weights, 'data', None, weights)
    # weights_data = None if weights is None
    # if weights_data is not None:
    if weights is not None:
        weights = weights[select]

    assert (len(x) == len(y)), "The length of input array doesn't match!"
    data_length = len(x)

    if x_left is None:
        x_left = min(x)
    if x_right is None:
        x_right = max(x)
    if y_left is None:
        y_left = min(y)
    if y_right is None:
        y_right = max(y)
    # x_left = attr.choose_value(kwargs, 'x_left', x, 'left', min, x)
    # y_left = attr.choose_value(kwargs, 'y_left', y, 'left', min, y)
    # x_right = attr.choose_value(kwargs, 'x_right', x, 'right', max, x)
    # y_right = attr.choose_value(kwargs, 'y_right', y, 'right', max, y)
    default_step_num = 40
    if x_step is None:
        x_step = (x_right - x_left) / default_step_num
    if x_window is None:
        x_window = x_step
    if y_step is None:
        y_step = (y_right - y_left) / default_step_num
    if y_window is None:
        y_window = y_step
    """
    x_step = attr.choose_value(kwargs, 'x_step', x, 'step',
                               (x_right - x_left) / default_step_num)
    y_step = attr.choose_value(kwargs, 'y_step', y, 'step',
                               (y_right - y_left) / default_step_num)
    x_window = attr.choose_value(kwargs, 'x_window', x, 'window', x_step)
    y_window = attr.choose_value(kwargs, 'y_window', y, 'window', y_step)
    """

    if z is not None:
        x_edges = np.arange(x_left, x_right + x_step, x_step)
        y_edges = np.arange(y_left, y_right + y_step, y_step)
        xbins = len(x_edges) - 1
        ybins = len(y_edges) - 1

        x_window_left = x_edges[:-1] + (x_step - x_window) / 2.
        y_window_left = y_edges[:-1] + (y_step - y_window) / 2.
        x_window_right = x_window_left + x_window
        y_window_right = y_window_left + y_window
    else:
        # calc hist
        # hist need a larger binning, and window is often larger than step
        at_least = 0  # avoid nan in bin_map if calc hist
        x_edges = np.arange(x_left, x_right + x_window, x_window)
        y_edges = np.arange(y_left, y_right + y_window, y_window)
        xbins = len(x_edges) - 1
        ybins = len(y_edges) - 1

        if weights is None:
            binned_map, x_edges, y_edges = np.histogram2d(
                x, y, bins=(x_edges, y_edges),
                range=[[x_left, x_right], [y_left, y_right]]
            )
            binned_map = binned_map.T
            return binned_map, x_edges, y_edges
        else:
            # normalize weights for calculating counts
            # weights = weights / mean(weights)
            x_window_left = x_edges[:-1]
            y_window_left = y_edges[:-1]
            x_window_right = x_edges[1:]
            y_window_right = y_edges[1:]
            z = np.ones_like(weights)

            def func(d, w):
                return np.nansum(d * w)

    """
    index_in_bin = np.empty((xbins, ybins), dtype=object)
    for i in np.ndindex(index_in_bin.shape):
        index_in_bin[i] = []
    x_bin_n = (x - x_left)/x_step
    x_bin_n = np.trunc(x_bin_n).astype(int)
    x_bin_n -= (x_bin_n == xbins).astype(int)
    y_bin_n = (y - y_left)/y_step
    y_bin_n = np.trunc(y_bin_n).astype(int)
    y_bin_n -= (y_bin_n == ybins).astype(int)
    for i in range(length):
        index_in_bin[x_bin_n[i]][y_bin_n[i]].append(i)
    """
    index_in_x_bin = np.zeros([xbins, data_length], dtype=bool)
    index_in_y_bin = np.zeros([ybins, data_length], dtype=bool)
    for i in range(xbins):
        index_in_x_bin[i] = x > x_window_left[i]
        index_in_x_bin[i] &= x <= x_window_right[i]
    for j in range(ybins):
        index_in_y_bin[j] = y > y_window_left[j]
        index_in_y_bin[j] &= y <= y_window_right[j]

    binned_map = np.zeros([xbins, ybins])
    # with alive_bar(xbins * ybins) as bar:
    with ProgressBar(xbins * ybins) as bar:
        for i in range(xbins):
            for j in range(ybins):
                index_in_xy_bin = index_in_x_bin[i] & index_in_y_bin[j]
                binned_map[i][j] = value_in_bin(z, index_in_xy_bin,
                                                weights=weights, at_least=at_least, func=func, bar=bar)

    # transpose the matric to follow a Cartesian convention. The same operation is NOT done in, e.g., np.histogram2d.
    binned_map = binned_map.T
    return binned_map, x_edges, y_edges


# %% func: hist
def hist(x, density=False, norm=False, **kwargs):
    """
    density: scaled by the step length
    norm: further scaling to make integrated area = 1

    kwargs:
        bins=100,
        x_left=min(x),
        x_right=max(x),
        x_step

    Returns
    -------
    hist, hist_err, bin_center, kwargs
    """
    x_left = attr.choose_value(kwargs, 'x_left', x, 'left', min, x)
    x_right = attr.choose_value(kwargs, 'x_right', x, 'right', max, x)
    select_list = kwargs.get('select_list')
    select, _ = attr.integrate_select(select_list)
    x = x[select]
    weights = kwargs.get('weights')
    if kwargs.get('x_step', None) is None:
        bins = kwargs.get('bins', 100)
        bin_edges = np.linspace(x_left, x_right, bins + 1)
        x_step = bin_edges[1] - bin_edges[0]
    else:
        x_step = kwargs.get('x_step')
        bin_edges = np.arange(x_left, x_right + x_step, x_step)
        bins = len(bin_edges) - 1
    if weights is None:
        hist, bin_edges = np.histogram(
            x, bins=bin_edges,
            range=(x_left,
                   x_right),
            density=False
        )
        hist_err = np.sqrt(hist)
    else:
        weights = weights[select]
        hist_ones = np.ones_like(weights)

        digitized_x = np.digitize(x, bin_edges)
        index_in_bin = [(digitized_x == i) for i in range(1, bins + 1)]
        hist = [value_in_bin(hist_ones, index_in_bin_i, weights=weights,
                             func=lambda d, w: np.nansum(d * w))
                for index_in_bin_i in index_in_bin]
        hist_err = [value_in_bin(hist_ones, index_in_bin_i, weights=weights,
                                 func=lambda d, w:
                                     np.sqrt(np.nansum(w ** 2)))
                    for index_in_bin_i in index_in_bin]

    hist = np.array(hist)
    hist_err = np.array(hist_err)
    scale = 1.
    if density:
        scale /= x_step
    if norm:
        scale /= np.nansum(hist)
    hist = hist * scale
    hist_err = hist_err * scale

    bin_center = (bin_edges[:-1] + bin_edges[1:]) / 2

    return bin_center, hist, hist_err, kwargs


# %% func: praction
def praction(select, within, weights=None):
    """
    TODO : use collapsed select list
    """
    if weights is None:
        return np.nansum(select & within) / np.nansum(within)
    else:
        return np.nansum(weights[select & within]) / np.nansum(weights[within])


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
