"""
Notes

I don't create a new class, because:
- Sometimes the modifying of the Table/DF will create a new object and destroy
  what I have created.
- Both Table and DF cannot recall the parent object within the columns, thus it
  is unconvenient when plotting because the information from the parent table
  cannt be read.

I don't use `attr.init` to give the basic metadata to each column because new
rows may be created without calling the `init`.

The `.name` attribute of columns is independent of `meta` and need to be handled independently.

If in the future I want to switch between astropy and pandas, I may write an
`attr.get` function to change the basic method without changing other codes.

Reload of the class DataSeries will disrupt the type check already loaded.

`name` is used in column name, file naming, title of selection, and the
default axis label, and should be non-mathematical string.  `label` is
used only in axis labels, and can be mathematical.
"""

# %% import
from functools import wraps

import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table, Column
from astropy.table.column import MaskedColumn
from numpy.lib.recfunctions import structured_to_unstructured
# import pandas as pd


# %% array & table
def table2array(table):
    """
    Turning a 2d table to array. Without this, the columns in the table cannot be merged normally.
    """
    if isinstance(table, Table):
        return structured_to_unstructured(table.as_array())
    else:
        return np.array(table)


def array2column(array: np.ndarray | Table,
                 meta_from: Table | Column = None,
                 **set_attrs) -> Column:
    """
    Turning an np.array or Table to Column, may with meta copy from somewhere, or set new attrs.

    Parameters
    ----------
    meta_from : has .meta to copy from

    **set_attrs : kwargs
        To be added into the meta of the array. May with 'name'.
    """

    if isinstance(array, Table):
        # avoid the index mixing from Table(list)
        array = table2array(array)
    elif not isinstance(array, np.ndarray):
        array = np.array(array)

    if not isinstance(meta_from, (Table, Column)):
        meta_from = None

    name = set_attrs.pop('name', None)
    if name is None:
        if meta_from is not None:
            name = meta_from.name
        else:
            if hasattr(array, 'name'):
                name = array.name
            else:
                name = 'col'
    try:
        column = Column(array, name=name)
    except TypeError:
        column = MaskedColumn(array, name=name)
    if meta_from is not None:
        column.meta.update(meta_from.meta)

    column.meta.update(set_attrs)
    return column
    """
    elif array.ndim == 2:
        table = Table(array)
        if meta_from is not None:
            table.meta.update(meta_from.meta)

        table.name = name
        table.meta.update(set_attrs)
        return table
    """


# %% set and reset attr
def set(table_or_column, **set_attrs):
    table_or_column.meta.update(set_attrs)


def reset(table_or_column, attr_name):
    return table_or_column.meta.pop(attr_name, None)
    """
    if table_or_column.meta.get(attr_name) is not None:
        return table_or_column.meta.pop(attr_name)
    else:
        return None
    """


def reset_limit(table_or_column):
    left = reset(table_or_column, 'left')
    right = reset(table_or_column, 'right')
    return left, right


"""
def set_limit(table_or_column, left=None, right=None):
    if left is not None:
        table_or_column.meta['left'] = left
    if right is not None:
        table_or_column.meta['right'] = right
"""


# %% sift_df
"""
def sift_df(table_or_column, min_=None, max_=None):
    if min_ is not None:
        table_or_column.mask(table_or_column <= min_, inplace=True)
    if max_ is not None:
        table_or_column.mask(table_or_column >= max_, inplace=True)
    return table_or_column
"""


# %% sift
"""
def _is_comparable(list_to_compare, threshold):
"""
# Return True if threshold is float, or int, or a list with the same dimension with list_to_compare.
"""
    if type(threshold) == float:
        return True
    if type(threshold) == int:
        return True
    if len(threshold) == len(list_to_compare):
        return True
    return False
"""


def _compare(list_to_compare, threshold, min_or_max):
    """
    Return a list of indices of the ones to be dropped compared to the threshold.

    Parameters
    ----------
    threshold: can be a float, or a list with the same dimension
    """

    drop_list = np.zeros_like(list_to_compare, dtype=bool)
    if threshold is None:
        return drop_list
    """
    name = 'max' if min_or_max == 1 else 'min'
    if _is_comparable(list_to_compare, threshold):
        threshold = [threshold]
    for threshold_i in threshold:
        if not _is_comparable(list_to_compare, threshold_i):
            raise Exception(f"The {name} is not comparale to the list.")
        drop_list = drop_list | (list_to_compare * min_or_max >= threshold_i * min_or_max)
    """
    drop_list = drop_list | (list_to_compare * min_or_max > threshold * min_or_max)
    return drop_list


def sift(original: np.ndarray | Column | MaskedColumn,
         min_: float | None = None,
         max_: float | None = None,
         use_list: np.ndarray | None = None,
         inplace: bool = True,
         doplot: bool = False,
         name: str = 'data'):
    """
    Filter with respect to min and max, while keeping the original dimension. The equal values are kept.

    The dropped items are replaced by np.nan. If the original array is int or bool, turn to float.

    Can plot the histograms before and after the filtering if doplot is set to True.

    Parameters
    ----------
    original: array
    min_, max_: can be a float, or a list with the same dimension.
    use_list: a bool list with the same dimension.
    inplace: True if I want the original array to be replaced.
    """
    original_arr = table2array(original)
    if original_arr.dtype == int or original_arr.dtype == bool:
        # int or bool array will turn into float if np.nan is added
        import warnings
        warnings.warn("Turning int or bool array into float! "
                      "You may want to use np.isclose(a, b) when comparing them.")
        original[:] = original.astype(float)

    is_masked = np.ma.is_masked(original_arr)
    if is_masked:
        drop_list = original_arr.mask
    else:
        drop_list = np.zeros_like(original_arr, dtype=bool)
    if use_list is not None:
        """
        if _is_comparable(original_arr, use_list):
            use_list = [use_list]
        for use_list_i in use_list:
            if not _is_comparable(original_arr, use_list_i):
                raise Exception(f"The {name} is not comparale to the list.")
            drop_list = drop_list | ~use_list_i
        """
        drop_list = drop_list | ~use_list

    drop_list = drop_list | _compare(original_arr, max_, 1)
    drop_list = drop_list | _compare(original_arr, min_, -1)
    cleared = np.where(drop_list, np.nan, original_arr)

    if doplot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 1.5), tight_layout=True)
        fig.suptitle(name)
        ax1.hist(original, bins=50, density=True)
        ax2.hist(cleared, bins=50, density=True)
        plt.show()

    cleared = array2column(cleared, meta_from=original)
    if inplace:
        # magic line: in-place update of the original array, while keeping all the metadata!
        original[:] = cleared
    return cleared


# %% func: choose_value
def choose_value(kwargs: dict,
                 xyz_attr_str: str,
                 xyz: Column | None,
                 attr_str: str,
                 func_or_default=None,
                 *func_args):
    """
    In a parent function, choose one of the values with different priorities.
    Update it in `kwargs` and return the value.
    Priorities:
        1st: use the one set in **kwargs (like kwargs['x_left']) when the function is called,
        2nd: None if xyz is None,
        3rd: use the attribute in xyz.meta,
        4th: use the default value, to get which a function calculation may be required,
        5th: None.

    Examples
    --------
    # use kwargs['x_step'], or x.meta['step'], or 0.1, and update kwargs['x_step']:
    choose_value(kwargs, 'x_step', x, 'step', 0.1)

    # use x.meta['edges'], or np.arange(x_left, x_right + x_step, x_step),
    # with no update in kwargs:
    x_edges = choose_value(kwargs, None, x, 'edges',
                           np.arange, x_left, x_right + x_step, x_step)

    Parameters
    ----------
    kwargs: the one to be updated.
    xyz_attr_str:
        The highest priority.
        The value set in **kwargs when the function is called.
        Should be like 'x_step'.
        Should be set to None if this priority is not considered.

    xyz: can be either np.array or Column, e.g. x, y, sfr
    attr_str: the name of an attribute from xyz.meta, e.g. 'label', 'left'
    func_or_default: the default value, or a function to calc the defalut value. `None` if no default is to be set.
    func_args: a set of args passed to func_to_calc_default when a further calculation is required.
    """

    xyz_attr_str_not_none = xyz_attr_str is not None
    xyz_attr_is_in_kwargs = xyz_attr_str_not_none and (kwargs.get(xyz_attr_str) is not None)
    # The above `and` is to protect because kwargs.get(None) may give non-None results

    def _update_kwargs_and_return(attr_value):
        if xyz_attr_str_not_none:
            kwargs[xyz_attr_str] = attr_value
        return attr_value

    # 1st prior: from kwargs
    if xyz_attr_is_in_kwargs:
        return kwargs.get(xyz_attr_str)
    # 2nd prior: None if xyz is None
    if xyz is None:
        return _update_kwargs_and_return(None)
    # 3rd prior: from column meta
    if hasattr(xyz, 'meta') and xyz.meta.get(attr_str) is not None:
        attr_value = xyz.meta[attr_str]
        return _update_kwargs_and_return(attr_value)
    # 5th prior: None
    if func_or_default is None:
        return _update_kwargs_and_return(None)
    # 4th prior: calc default
    if callable(func_or_default):
        return _update_kwargs_and_return(func_or_default(*func_args))
    return _update_kwargs_and_return(func_or_default)


# %% wrap: set values
def get_name(xyz, xyz_str, to_latex=False):
    """
    If x itself is a str, return x.
    Second priority    : return x.name like 'mstar'
    If x has no `name`: return xyz_str like 'x', or r'$X$' if to_latex is set
    """
    if isinstance(xyz, str):
        return xyz
    if hasattr(xyz, 'name') and xyz.name is not None:
        return xyz.name
    else:
        if to_latex:
            return rf'${xyz_str.upper()}$'
        return xyz_str


def get_default(xyz, xyz_str, attr_str, kwargs, set_default):
    """for one attr like 'x_left', refresh kwargs using `choose_value`
    xyz is like x, y, z
    xyz_str is like 'x', 'y', 'z'
    attr_str is like 'left', 'right', 'label'
    set_default is bool
    """
    # import inside the function to avoid circular import
    from . import calc

    if set_default:
        if attr_str == 'left':
            default_args = (calc.min, xyz)
        elif attr_str == 'right':
            default_args = (calc.max, xyz)
        elif attr_str == 'label':
            default_label = get_name(xyz, xyz_str, to_latex=True)
            default_args = [default_label]
        else:
            default_args = []
    else:
        default_args = []

    choose_value(kwargs,
                 '_'.join([xyz_str, attr_str]),  # resume 'x_left'
                 xyz, attr_str,
                 *default_args)
    return kwargs


def set_values(to_set=None, set_default=False):
    """
    to_set is like ['x_step', 'y_label']
    Update the kwargs with the value selected between kwargs['x_step'], x.meta['step'],
    or the default value if `set_default` is True.
    When using the decorator, the function may have args like `x_left=None` explicitly set, or hidden in **kwargs.
    If y or z is None, the calculation is automatically skipped through the choose_value function.
    Only left, right and label are supported to set default values.
    """
    def decorate(called_func):
        @wraps(called_func)
        def set_values_core(x, y=None, z=None, *args, **kwargs):
            if to_set is not None:
                # to_set_split is like [['x', 'step'], ['y', 'label']]
                to_set_split = [i.split('_') for i in to_set]
                # clean up to_set_split, remove the ones with wrong length and leave the ones like ['x', 'step']
                to_set_like_xyz_attr = [i for i in to_set_split if len(i) == 2]
                # for each of them, refresh the kwargs
                for xyz_attr_list in to_set_like_xyz_attr:
                    # xyz_attr_list is like ['x', 'step']
                    xyz_str = xyz_attr_list[0]  # like 'x'
                    xyz = eval(xyz_str)  # like x
                    attr_str = xyz_attr_list[1]  # like 'step'
                    # refresh kwargs
                    get_default(xyz, xyz_str, attr_str, kwargs, set_default)
            # now that all the kwargs are refreshed, call the function
            if y is None:
                return called_func(x, *args, **kwargs)
            elif z is None:
                return called_func(x, y=y, *args, **kwargs)
            else:
                return called_func(x, y=y, z=z, *args, **kwargs)
        return set_values_core
    return decorate


# %% class: DataFrameM
"""
class DataFrameM(pd.DataFrame):
    # if the data is a 2d map, use `z_map = DFM({'map': list(z_map)})`
    def get_attr(self, colname, attr_name):
        return self[colname].meta[attr_name]

    def set(self, attr_name, colname, value):
        self[colname].meta[attr_name] = value

    def set_label(self, colname, label):
        self[colname].meta['label'] = label

    def set_limit(self, colname, left=None, right=None):
        if left is not None:
            self[colname].meta['left'] = left
        if right is not None:
            self[colname].meta['right'] = right

    def set_critical_value(self, colname, critical_value):
        self[colname].meta['critical_value'] = critical_value

    def set_step(self, colname, step):
        self[colname].meta['step'] = step
        if self[colname].meta.get('window') is None:
            self[colname].meta['window'] = step

    def set_window(self, colname, window):
        self[colname].meta['window'] = window
        if self[colname].meta.get('step') is None:
            self[colname].meta['step'] = window

    # def copy(self):
        # return copy.deepcopy(self)

    # def reset_data(self, colname, newdata, keep_limit=False, **kwargs):
        # This will not change the original data, but return a new one.
        # newself = self.copy()
        # newself[colname].meta['data'] = self.sift(newdata, **kwargs)
        # if not keep_limit:
        # newself[colname].meta['left'] = calc.min(newdata)
        # newself[colname].meta['right'] = calc.max(newdata)
        # return newself

    def copy_meta(self, from_, to_=0):
        # to_=slice(None) corresponds to the whole frame
        # for single-column frame, use
            # y_edges = DFM(y_edges)
            # y_edges = y_edges.copy_meta(y)
        if isinstance(from_, str):
            self[to_].meta = self[from_].meta
        else:
            self[to_].meta = from_.meta
        return self[to_]

    def reset_limit(self, colname):
        # if self[colname].meta.get('right') is not None:
        self[colname].meta.pop('right', None)
        # if self[colname].meta.get('left') is not None:
        self[colname].meta.pop('left', None)
"""
