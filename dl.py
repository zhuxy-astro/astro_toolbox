#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# %% import
import requests
import os
from astropy.table import Table
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from multiprocessing.pool import ThreadPool
from functools import partial

# unify the usage of progress bar
# using the astropy progress bar will partly mess up the output in multiprocessing
try:
    from alive_progress import alive_bar
    progressbar = alive_bar

    def update(bar):
        bar()

except ImportError:
    from astropy.utils.console import ProgressBar
    progressbar = ProgressBar

    def update(bar):
        bar.update()


# %% class Naming
class Naming:
    """set rules for one specific naming method
    """
    def __init__(self, col_name_list, string_format):
        self.col_name_list = col_name_list
        self.string_format = string_format

    def get(self, table_row):
        """when a rule is set, return the name for a specific row, or raise ValueError.
        The error raised here only means that the row doesn't fit this specific rule, but other rules may apply.
        So the error is catched in _set_naming_priority.
        """
        value_list = []
        for col_name in self.col_name_list:
            if col_name not in table_row.colnames:
                raise ValueError(f'No valid value for {col_name}.')
            value = table_row[col_name]
            no_valid_value = (value is None) or np.ma.is_masked(value)
            if not no_valid_value:
                # isreal is to protect isnan from str
                no_valid_value = no_valid_value or (np.isreal(value) and np.isnan(value))
            if no_valid_value:
                raise ValueError(f'No valid value for {col_name}.')
            value_list.append(table_row[col_name])
        return self.string_format.format(*value_list)


# %% class _SurveyBase
class _SurveyBase:
    """Base class for surveys.
    The setting of table is not necessary when using show_fig or get_url.
    The ra and dec hould be in degrees, and are renamed by default.
    When initiallizing a new survey, set the columns to be renamed and the required columns.
    """
    def _rename_col(self):
        """convert possible variations of column names to a standard name
        OK if there is no match.
        """
        for new_name, possible_names in self._rename_dict.items():
            for name in possible_names:
                if name in self._table.colnames:
                    self._table.rename_column(name, new_name)
                    break

    def _check_required_cols(self):
        """check if the table has all required columns
        """
        assert set(self._required_cols).issubset(self._table.colnames), \
            'table should include: ' + ', '.join(self._required_cols)

    def __init__(self, table=None, required_cols=[], rename_dict={}):
        """table could be an astropy table or a row
        """
        self._TIMEOUT = 30  # seconds. Different surveys may need different timeout
        self._THREADS = 50  # number of threads for downloading images
        self._FILE_SUFFIX = '.jpg'
        self._required_cols = required_cols
        # rename ra and dec by default
        self._rename_dict = {'ra': ['ra', 'RA'], 'dec': ['dec', 'DEC', 'Dec']}
        self._rename_dict.update(rename_dict)
        # RA and Dec as default naming method
        self.name_ra_dec = Naming(['ra', 'dec'], 'RA{0:.4f}, Dec{1:.4f}')

        if table is None:
            self._table = None
            return

        # with table is available
        if isinstance(table, Table):
            self._table = table.copy()  # copying is necessary to avoid changing the original table
        else:
            self._table = Table(table)

        self._rename_col()
        self._check_required_cols()

    def _set_naming_priority(self, table_row, *nameing_methods):
        """used to define naming seq for different surveys.
        """
        for naming_method in nameing_methods:
            try:
                return naming_method.get(table_row)
            except ValueError:
                if naming_method is nameing_methods[-1]:
                    raise ValueError(f'No valid naming method for {table_row}')
                continue

    def naming_seq(self, table_row):
        """the default naming is RA and Dec
        """
        return self._set_naming_priority(table_row, self.name_ra_dec)

    def get_url(self, table_row, *args, **kwargs):
        """always different for different surveys
        """
        raise NotImplementedError

    def _get_response(self, table_row):
        url = self.get_url(table_row)
        response = requests.get(url, stream=True, timeout=self._TIMEOUT)
        return response

    def show_fig(self, table_row, timeout=None, naming_seq=None):
        """show the image in plt.
        Setting of table is not necessary when using this function.
        """
        if timeout is not None:
            self._TIMEOUT = timeout
        if naming_seq is None:
            naming_seq = self.naming_seq

        response = self._get_response(table_row)
        img = Image.open(response.raw)

        fig = plt.figure(figsize=(6.4, 4.8))
        ax = fig.add_subplot(111)
        ax.set_title(naming_seq(table_row),
                     fontfamily='sans-serif', fontsize=16)
        ax.imshow(img)
        ax.axis('off')
        # ax.set_xticks([])
        # ax.set_yticks([])
        fig.show()

    def _get_one_fig(
        self, table_row, failed_table, invalid_table,
        savedir, naming_seq,
        overwrite, bar,
    ):
        """download and write one image
        putting try inside this function can avoid error raising in multitasking
        """
        if naming_seq is None:
            naming_seq = self.naming_seq
        try:
            filename = naming_seq(table_row) + self._FILE_SUFFIX
        except ValueError:
            invalid_table.add_row(table_row)
        savepath = os.path.join(savedir, filename)
        if not overwrite and os.path.exists(savepath):
            update(bar)
            return

        try:
            response = self._get_response(table_row)
            # TODO: when to raise? If the pic is empty but still returned? What is the good codes?
            response.raise_for_status()
            fig = response.content
            with open(savepath, 'wb') as f:
                f.write(fig)
            update(bar)
        except ValueError:
            # TODO: this is not raised.
            invalid_table.add_row(table_row)
        except Exception:
            # TODO: will this catch the above one?
            failed_table.add_row(table_row)

    def _get_figs_once(self, task_table, failed_table, invalid_table, **kwargs):
        """download and write multiple images using multiple threads
        everything in kwargs will be passed to _get_one_fig
        """
        pool = ThreadPool(self._THREADS)
        func = partial(self._get_one_fig,
                       failed_table=failed_table, invalid_table=invalid_table,
                       **kwargs)
        pool.map(func, task_table)
        pool.close()
        pool.join()
        return failed_table, invalid_table

    def get_figs(self, savedir='img', overwrite=False,
                 naming_seq=None, try_loops=3, timeout=None):
        """if naming_seq is None, use the default naming sequence for this survey
        TODO: doc on the params
        """
        if timeout is not None:
            self._TIMEOUT = timeout

        if not os.path.exists(savedir):
            os.mkdir(savedir)

        task_table = self._table.copy()

        with progressbar(len(task_table)) as bar:
            try_i = 0
            for try_i in range(try_loops):
                num_left = len(task_table)
                if num_left == 0:
                    break
                print(f"Try {try_i}. {num_left} images to download ...\n")
                failed_table = Table(names=self._table.colnames, dtype=self._table.dtype)
                invalid_table = Table(names=self._table.colnames, dtype=self._table.dtype)
                # the failed table will be used as the new task table
                task_table, invalid_table = self._get_figs_once(
                    task_table, failed_table, invalid_table,
                    savedir=savedir, naming_seq=naming_seq,
                    overwrite=overwrite,
                    bar=bar,
                )

        print(f'Done. {len(failed_table)} items failed, {len(invalid_table)} items invalid.')
        return failed_table, invalid_table


# %% class SDSS
class SDSS(_SurveyBase):
    def _get_plate_mjd_fiber_name(self, table_row):
        name = f"{table_row['plate']},{table_row['mjd']},{table_row['fiber']}"
        return name

    def __init__(self, *args, **kwargs):
        _SDSS_RENAME_DICT = {
            'plate': ['plate', 'plateid', 'plate_id', 'PLATE', 'PLATEID', 'PLATE_ID'],
            'mjd': ['mjd', 'MJD'],
            'fiber': ['fiber', 'fiberid', 'fiber_id', 'FIBER', 'FIBERID', 'FIBER_ID'],
            'specobjid': ['specobjid', 'SPECOBJID', 'specobj_id', 'SPECOBJ_ID',
                          'spec_id', 'SPEC_ID', 'specid', 'SPECID'],
        }
        _SurveyBase.__init__(self, *args, rename_dict=_SDSS_RENAME_DICT, **kwargs)
        self._TIMEOUT = 15
        self.name_plate_mjd_fiber = Naming(['plate', 'mjd', 'fiber'], '{0}, {1}, {2}')
        if self._table is None:
            return

    def naming_seq(self, table_row):
        return self._set_naming_priority(table_row,
                                         self.name_plate_mjd_fiber,
                                         self.name_ra_dec)


class SDSSImg(SDSS):
    def __init__(self, *args, **kwargs):
        SDSS.__init__(self, *args, required_cols=['ra', 'dec'], **kwargs)
        if set(['plate', 'mjd', 'fiber']).issubset(self._table.colnames):
            self._get_name = self._get_plate_mjd_fiber_name

    def get_url(self, table_row, dr='dr17', scale=0.2, length=200, opt=''):
        """
        `opt` sets the mark on the image. 'G' for galaxy with crosses, 'S' for star with a red square.
        example:
        https://skyserver.sdss.org/dr17/SkyServerWS/ImgCutout/getjpeg?TaskName=Skyserver.Explore.Image&ra=195.299786348959%20&dec=29.6641012044628&scale=0.2&width=200&height=200&opt=G
        """
        ra = table_row['ra']
        dec = table_row['dec']
        url = (
            f"https://skyserver.sdss.org/{dr}/SkyServerWS/ImgCutout/getjpeg?TaskName=Skyserver.Explore.Image&"
            f"ra={ra}%20&dec={dec}&scale={scale}&width={length}&height={length}&opt={opt}")
        return url


class SDSSSpec(SDSS):
    def __init__(self, *args, **kwargs):
        SDSS.__init__(self, *args, required_cols=['specobjid'], **kwargs)
        self.name_specobjid = Naming(['specobjid'], 'specid{0}')

    def naming_seq(self, table_row):
        return (self._set_naming_priority(table_row,
                                          self.name_plate_mjd_fiber,
                                          self.name_ra_dec,
                                          self.name_specobjid,
                                          )
                + 'spec')

    def get_url(self, table_row, dr='dr17'):
        """
        example:
        https://skyserver.sdss.org/dr17/en/get/SpecById.ashx?id=320932083365079040
        """
        specobjid = table_row['specobjid']
        url = f"https://skyserver.sdss.org/{dr}/en/get/SpecById.ashx?id={specobjid}"
        return url


# %% class DESI
class DESIImg(_SurveyBase):
    def __init__(self, *args, **kwargs):
        _SurveyBase.__init__(self, *args, required_cols=['ra', 'dec'], **kwargs)

    def get_url(self, table_row, layer='ls-dr9', bands='grz', pixscale=0.2, size=200):
        """
        example:
        """
        ra = table_row['ra']
        dec = table_row['dec']
        url = (
            "https://www.legacysurvey.org/viewer/cutout.jpg?"
            f"ra={ra}&dec={dec}&layer={layer}&pixscale={pixscale}&bands={bands}&size={size}")
        return url
