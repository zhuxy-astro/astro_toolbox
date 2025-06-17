#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# %% import
import numpy as np
from astropy.table import Column

from . import attr, calc


# %% combine selections
def _status_of_list_of_selects(list_of_selects, reference=None):
    """if list_of_selects is empty, return 0;
    if it is a single selection, return 1;
    if it is a list of selections, return 2.
    """

    # the comparison between Column and slice(None) is an array of bool
    list_of_selects_is_empty = (list_of_selects == slice(None))
    if hasattr(list_of_selects_is_empty, 'any'):
        list_of_selects_is_empty = list_of_selects_is_empty.any()

    list_of_selects_is_empty = (
        list_of_selects_is_empty
        or (list_of_selects is None)
        or (len(list_of_selects) == 0))

    if list_of_selects_is_empty:
        return 0

    if reference is not None:
        list_of_selects_is_single = len(list_of_selects) == len(reference)
    else:
        list_of_selects_is_single = np.ndim(list_of_selects) <= 1

    if list_of_selects_is_single:
        return 1

    return 2


def create_select_all(reference=None):
    if reference is not None:
        # TODO: reference may be masked!
        return Column(np.ones(len(reference), dtype=bool), name='')
    else:
        return slice(None)  # note that no name is returned here


def combine_names(list_of_selects, reference=None, list_status=None):
    """return the connection of names joined by ', '
    """
    if list_status is None:
        list_status = _status_of_list_of_selects(list_of_selects, reference)

    if list_status == 0:
        return ''

    if list_status == 1:
        if hasattr(list_of_selects, 'name'):
            return list_of_selects.name
        else:
            return ''

    name_list = []
    for s in list_of_selects:
        if hasattr(s, 'name') and s.name != '':
            name_list += [s.name]
    name_combined = ', '.join(name_list)

    return name_combined


def combine(list_of_selects, reference=None):
    """
    Input a selection array or a list of selections, output the join of them in Column.
    The name of the output is set in Column.name.
    The `reference` is an array in the same dimension, used to determine whether the
        `list_of_selects` is single or multiple, and is recommended to be set.

    Returns
    -------
    select_result : an bool array with the combined name
    """
    list_status = _status_of_list_of_selects(list_of_selects, reference)

    if list_status == 0:
        return create_select_all(reference)

    elif list_status == 1:
        return attr.array2column(list_of_selects)

    select_result = np.ones_like(list_of_selects[0], dtype=bool)
    for s in list_of_selects:
        if isinstance(s, slice):
            mask = np.zeros_like(select_result, dtype=bool)
            mask[s] = True
            s = mask
        select_result &= s
    name_combined = combine_names(list_of_selects, reference, list_status)

    return attr.array2column(select_result, name=name_combined)


# %% cross selections
def cross(list_of_list_of_selects, base=None):
    if base is None:
        cross_selections = [create_select_all(base)]
    else:
        base = base.copy()
        base.name = ''
        cross_selections = [base]
    # for each dimension
    for i, list_of_selects in enumerate(list_of_list_of_selects):
        list_status = _status_of_list_of_selects(
            list_of_selects, reference=cross_selections[0])
        if list_status == 1:
            list_of_selects = [list_of_selects]

        cross_selections_last_round = cross_selections
        cross_selections = []
        # for each selection in the dimension
        for j, new_selection in enumerate(list_of_selects):
            for accumulated_selection in cross_selections_last_round:
                cross_selections.append(combine([
                    accumulated_selection, new_selection], reference=base))
    return cross_selections


# %% select good
def good(array):
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
    try:
        mask |= ~np.isfinite(array)
    except TypeError:
        pass
    return ~mask


# %% func: select range_ and select value_edges
def range_(data, left=None, right=None,
           name=None, label=None):
    """return a select array cut by one edge or two edges
    """
    if name is None:
        if hasattr(data, 'name'):
            name = data.name
        else:
            name = 'x'
    if label is None:
        if hasattr(data, 'meta'):
            label = data.meta.get('label', None)
    if label is None:  # still None
        label = name

    le = '<='
    le_math = r'\leq'

    if left is None:
        if right is None:
            raise ValueError('At least one of left and right should be set.')

        return attr.array2column(
            data <= right,
            right=right,
            name=f'{name}{le}{right:.3g}',
            label=rf'${label} {le_math} {right:.3g}$')

    if right is None:
        return attr.array2column(
            data > left,
            left=left,
            name=f'{name}>{left:.3g}',
            label=rf'${label} > {left:.3g}$')

    return attr.array2column(
        (data > left) & (data <= right),
        right=right, left=left,
        name=f'{left:.3g}<{name}{le}{right:.3g}',
        label=rf'${left:.3g} < {label} {le_math} {right:.3g}$')


def edges(data, edges, name=None, label=None):
    """return the list of select arrays cut by the edges, with length = len(edges) + 1
    data is a column
    """
    select_list = []

    select_list.append(range_(
        data, right=edges[0], name=name, label=label))

    for i in range(len(edges) - 1):
        select_list.append(range_(
            data, left=edges[i], right=edges[i + 1], name=name, label=label))

    select_list.append(range_(
        data, left=edges[-1], name=name, label=label))

    return select_list


# %% func: select id
def id(table, value, name=['plate', 'mjd', 'fiberid']):
    """
    can deal with one or multiple values and names, and multiple rows.
    returns a bool array
    When `name` has three attrs like ['p', 'm', 'f'],
        `value` is like [280, 51612, 97] or like [[280, 51612, 97], [996, 52641, 513]]
    When `name` has one attr like 'plate', or ['plate']
        `value` is like 280, or like [280, 996]
    """
    is_single_name = type(name) is str
    if not is_single_name and (len(name) == 1):
        is_single_name = True
        name = name[0]
    if is_single_name:
        if np.ndim(value) == 0:
            # single value
            return table[name] == value
        else:
            # multiple values
            return np.isin(table[name], value)

    # multiple name
    if len(value) == len(name):
        # single value
        value = [value]
    select_result = np.zeros(len(table), dtype=bool)
    for value_j in value:
        select_single = np.ones(len(table), dtype=bool)
        for name_i, value_ij in zip(name, value_j):
            select_single &= (table[name_i] == value_ij)
        select_result |= select_single
    return select_result


# %% func: select percentile
def percentile(x,
               percentile=[0.25, 0.50, 0.75],
               weights=None,
               select=slice(None)):
    """percentile must be a list or array
    """
    percentile = np.array(percentile)
    cuts = calc.weighted_percentile(
        data=x, weights=weights,
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
