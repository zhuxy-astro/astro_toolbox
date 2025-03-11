#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# %% import
import h5py
import numpy as np
from astropy.table import Table, MaskedColumn  # , hstack
from astropy.io import fits
# import pandas as pd


# %% func: read_hdf5
def hdf2table(filename, t=Table()):
    """
    Usage:
        from astropy.table import Table
        kai = read_hdf5('path/to/file.hdf5')
        kai = read_hdf5('path/to/another/file.hdf5', kai)
    """
    with h5py.File(filename, 'r') as f:
        """
        if len(f) == 1:
            key = list(f.keys())[0]
            new_table = Table(np.array(f[key]))
            t = hstack([t, new_table])
        else:
        """
        try:
            t = Table.read(f)
        except Exception:
            for key_name in f.keys():
                t[key_name] = f[key_name]

    # hdf5 files written by astropy will have `.mask` suffix for masked columns
    mask_suffix = '.mask'
    for name in t.colnames:
        if name.endswith(mask_suffix):
            name_clean = name.replace(mask_suffix, '')
            if name_clean in t.colnames:
                t[name_clean] = MaskedColumn(t[name_clean], mask=t[name])
                t.remove_column(name)

    # this should be done at reading the table with `character_as_bytes=False`, but it doesn't work.
    # manually converting bytes tring to unicode here
    t.convert_bytestring_to_unicode()
    return t


# %% func: generate col name list
def coln_list(n_list):
    return [f'col{i}' for i in n_list]


# %% func: select cols and rename
def select_col_rename(table, col_name_rename_dict):
    col_name = list(col_name_rename_dict)
    col_rename = [col_name_rename_dict[i] for i in col_name]
    col_name = [f'col{i}' if isinstance(i, int) else i for i in col_name]
    table_part = table[col_name]
    table_part.rename_columns(col_name, col_rename)
    return table_part


# %% func: io bewteen df and fits
# https://github.com/wkcosmology/enhanced/blob/master/tool/io.py
# written by Kai Wang
def data_frame2fits(filename, df, columns=None, indexs=None, units=None, comments=None):
    """write the data in pandas.DataFrame to a fits file

    Parameters
    ----------
    filename : str
        the output file name
    df : :class:pandas.DataFrame
        the dataframe contains the data
    columns : list, optional
        store the data into len(list) HDU blocks
    indexs : list, optional
        store the data into len(index) HDU blocks
    units : list, optional
        contains the unit for each column
    comments : dict, optional
        comments for the primaryHDU
    """
    if columns is None:
        columns = [list(df.columns)]
    if indexs is None:
        indexs = [np.arange(len(df))]
    assert not ((len(columns) > 1) and (len(indexs) > 1))
    if units is None or isinstance(units, str):
        units = [units] * max(len(columns), len(indexs))
    assert isinstance(units, list)

    hdr = fits.Header()
    if comments is not None:
        for k, v in comments.items():
            hdr[k] = v
    primary_hdu = fits.PrimaryHDU(header=hdr)
    hdu_list = [primary_hdu]

    if len(columns) > 1:
        for cs, us in zip(columns, units):
            hdu_list.append(_table2hdu_units(Table.from_pandas(df.iloc[indexs[0]][cs]), cs, us))
    else:
        for index, us in zip(indexs, units):
            hdu_list.append(_table2hdu_units(Table.from_pandas(df.iloc[index][columns[0]]), columns[0], us))

    hdul = fits.HDUList(hdu_list)
    hdul.writeto(filename)


def _table2hdu_units(table, colnames=None, units=None):
    """convert a astropy.table.Table object to astropy.io.fits.TableHDU

    Parameters
    ----------
    table : :class:pandas.DataFrame
        the dataframe contains the data
    units : list, optional
        the list of string contains the name of columns
    units : list, optional
        the list of string contains the unit of columns

    Returns
    -------
    :class:astropy.io.fits.PrimaryHDU
    """
    hdu = fits.table_to_hdu(table)
    if units is not None and colnames is not None:
        assert isinstance(units, list)
        for c, u in zip(colnames, units):
            fits.ColDefs(hdu).change_unit(c, u)
    return hdu


def fits2data_frame(filename, hdu=None):
    """read the pandas data frame from a fits file

    Parameters
    ----------
    filename : str
        the filepath and name of the fits file
    hdu : int
        index of HDU to read

    Returns
    -------
    :class: pandas.DataFrame
        the pandas dataframe in the fits file

    """
    f = fits.open(filename)
    if hdu is None and len(f) == 2:
        return Table.read(filename).to_pandas()
    elif hdu is None and len(f) >= 2:
        return [Table.read(filename, hdu=i).to_pandas() for i in range(1, len(f))]
    elif isinstance(hdu, int):
        return Table.read(filename, hdu=hdu).to_pandas()
    else:
        return [Table.read(filename, hdu=i).to_pandas() for i in hdu]
