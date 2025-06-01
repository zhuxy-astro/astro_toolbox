from astropy.table import Table
from astropy.table.column import MaskedColumn
import numpy as np


# %% get_formats_dict
def get_formats_dict(table, fmt='%.5g', exclude_cols=None):
    formats_dict = dict()
    if exclude_cols is None:
        exclude_cols = dict()
    for colname in table.colnames:
        if colname in exclude_cols:
            continue
        is_float = np.issubdtype(table.dtype[colname], np.floating)
        if is_float:
            formats_dict[colname] = fmt
    return formats_dict


# %% table cleaning
def clean_table(t: Table):
    """fill table with np.nan if it is masked and float
    """
    for name in t.colnames:
        col_is_float = np.issubdtype(t.dtype[name], np.floating)
        col_is_masked = isinstance(t[name], MaskedColumn)
        if (col_is_float and col_is_masked):
            t[name] = t[name].filled(np.nan)
    return t
