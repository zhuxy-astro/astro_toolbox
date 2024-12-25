#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# %% import
import os
import requests
from multiprocessing.pool import ThreadPool
from functools import partial

import numpy as np
from astropy.table import Table
import matplotlib.pyplot as plt
from PIL import Image

from .misc import Bar


# %% class Naming
class Naming:
    """set rules for one specific naming method
    """
    def __init__(self, col_name_list, string_format=''):
        # empty string_format is used when I just want to check the validity of the col_name_list in the table_row
        self.col_name_list = col_name_list
        self.string_format = string_format

    def get(self, table_row):
        """when a rule is set, return the name for a specific row, or raise ValueError.
        The error raised here only means that the row doesn't fit this specific rule, but other rules may apply.
        So the error is catched in set_naming_priority.
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
    The ra and dec should be in degrees.

    Usage:
    - Download one image and show in plt:
        `dl.SDSSImg.show_fig(table_row, timeout=None, naming_seq=None)`
        For a one-liner, use `dl.SDSSImg.show_fig(Table([[35.7], [-4.9]], names=['ra', 'dec'])[0])`
    - Get the url for one image:
        `dl.SDSSImg.get_url(table_row, dr='dr17', scale=0.2, length=200, opt='')`
    - Download multiple images:
        ```
        failed_table, invalid_table = dl.SDSSImg(table).get_figs(
            savedir='img', overwrite=False, naming_seq=None, suffix='',
            try_loops=3, timeout=None)
        ```
    If another naming sequence is needed, use it in the function arg like
    `naming_seq=dl.SDSSImg.naming_seq`

    When initiallizing a new survey:
        - The `_required_cols` (in a list) should be set before the __init__
        method.
        - Set the url for downloading images in the `get_url` method, and set
        it as a static method.
        - (Optional) Set the columns to be renamed (in a dict) when
        initializing. RA and Dec are renamed by default. `__init__` method
        does not need to be called again if no more renaming other than RA and
        Dec is needed.
        - (Optional) Set the naming sequence `naming_seq` using the
        `set_naming_priority` method. The default is RA and Dec only. Set it
        as a class method.
        - (Optional) If a new naming rule is needed, define it using the
        `Naming` class, and put it outside the `__init__` method.
        - (Optional) New `_timeout`, `_threads`, and `_extension` should also
        be set outside the `__init__` method. These are not to be changed
        within one survey.
    """
    def _rename_cols(self):
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

    # default parameters for all surveys
    _required_cols = []  # this is set outside of __init__ because classmethod set_naming_priority uses this

    # RA and Dec are the default naming method. It is defined in the class, not in the instance.
    name_ra_dec = Naming(['ra', 'dec'], 'RA{0:.4f}, Dec{1:.4f}')

    # not going to change this unless when defining a new survey
    _timeout = 30  # seconds. Different surveys may need different timeout
    _threads = 50  # number of threads for downloading images
    _extension = '.jpg'

    def __init__(self, table=None, rename_dict=None):
        """table could be an astropy table or a row
        `table` is given when the class is initialized,
        `rename_dict` is given when a survey child class is defined.
        """
        # rename ra and dec by default
        if rename_dict is None:
            rename_dict = dict()
        self._rename_dict = {'ra': ['ra', 'RA'], 'dec': ['dec', 'DEC', 'Dec']}
        self._rename_dict.update(rename_dict)

        if table is None:
            self._table = None
            return

        # with table is available
        if isinstance(table, Table):
            self._table = table.copy()  # copying is necessary to avoid changing the original table
        else:
            self._table = Table(table)

        self._rename_cols()
        self._check_required_cols()

    @classmethod
    def set_naming_priority(cls, table_row, naming_methods_list, required_cols=None):
        """used to define naming seq for different surveys.
        returns a string or raises ValueError.
        """
        if required_cols is None:
            required_cols = cls._required_cols
        # if the row doesn't have all required_cols, ValueError will be raised
        _ = Naming(required_cols).get(table_row)

        for naming_method in naming_methods_list:
            try:
                return naming_method.get(table_row)
            except ValueError:
                if naming_method is naming_methods_list[-1]:  # the last try
                    raise ValueError(f'No valid naming method for {table_row}')
                # during the loop, continue to try names
                continue

    @classmethod
    def naming_seq(cls, table_row):
        """the default naming is RA and Dec.
        This is a class method because it reads the naming rules from the class.
        returns a string or raises ValueError following `set_naming_priority`.
        """
        return cls.set_naming_priority(table_row, [cls.name_ra_dec])

    @staticmethod
    def get_url(table_row, *args, **kwargs):
        """always different for different surveys
        """
        raise NotImplementedError

    @classmethod
    def _get_response(cls, table_row, timeout=None):
        """this is called by show_fig, which is a class method. Therefore this is a class method, too.
        The reason not to make it static is that it has to read the timeout, which may differ between surveys.
        """
        if timeout is None:
            timeout = cls._timeout
        url = cls.get_url(table_row)
        response = requests.get(url, stream=True, timeout=timeout)
        response.raise_for_status()
        return response

    @staticmethod
    def validate_img(response):
        """check if the image is valid, if true, return the image.
        """
        try:
            img = Image.open(response.raw)
        except Image.UnidentifiedImageError:
            raise ValueError('No valid image returned.')
        return img

    @classmethod
    def fig_img(cls, table_row, timeout=None):
        """return the image as a PIL image object, which can be shown in plt.imshow(img)
        Setting of table is not necessary when using this function.
        """
        response = cls._get_response(table_row, timeout=timeout)
        img = cls.validate_img(response)
        return img

    @classmethod
    def show_fig(cls, table_row, timeout=None, naming_seq=None):
        """show the image in plt
        """
        img = cls.fig_img(table_row, timeout=timeout)

        if naming_seq is None:
            naming_seq = cls.naming_seq

        fig = plt.figure(figsize=(6.4, 4.8))
        ax = fig.add_subplot(111)
        ax.set_title(naming_seq(table_row),
                     fontfamily='sans-serif', fontsize=16)
        ax.imshow(img)
        ax.axis('off')
        fig.show()

    def _get_one_fig(
        self, table_row, failed_table, invalid_table,
        save_in_list,
        savedir, naming_seq, suffix,
        overwrite, timeout, bar,
    ):
        """download and write one image
        putting try inside this function can avoid error raising in multitasking
        """
        if save_in_list is None:
            if naming_seq is None:
                naming_seq = self.naming_seq

            try:
                filename = naming_seq(table_row) + suffix + self._extension
            except ValueError:  # no valid name
                invalid_table.add_row(table_row)
                bar()
                return

            savepath = os.path.join(savedir, filename)
            if not overwrite and os.path.exists(savepath):
                bar()
                return

        # TODO: sometimes rows may fail but not in the list
        try:
            response = self._get_response(table_row, timeout=timeout)
            # _ = self.validate_img(response)  # checking validation using PIL causes error
            # it might be possible to check if the image is empty, but as SDSS
            # images have words on empty images, it is not easy to check.
            if save_in_list is not None:
                save_in_list[table_row['index']] = self.validate_img(response)
            else:
                img_content = response.content
                with open(savepath, 'wb') as f:
                    f.write(img_content)
            bar()
        except ValueError:  # no valid image, often when the http returns something non-image
            invalid_table.add_row(table_row)
            bar()
        except Exception:  # as err:  # other errors
            # print(err)
            # this will not catch the above ValueError
            failed_table.add_row(table_row)

    def _get_figs_once(self, task_table, failed_table, invalid_table, **kwargs):
        """download and write multiple images using multiple threads
        everything in kwargs will be passed to _get_one_fig
        """
        # TODO: how to quit?
        pool = ThreadPool(self._threads)
        func = partial(self._get_one_fig,
                       failed_table=failed_table, invalid_table=invalid_table,
                       **kwargs)
        pool.map(func, task_table)
        pool.close()
        pool.join()
        return failed_table, invalid_table

    def get_figs(self,
                 save_in_list=None,
                 savedir='img', overwrite=False, naming_seq=None, suffix='',
                 try_loops=3, timeout=None):
        """if naming_seq is None, use the default naming sequence for this survey
        `suffix` is before the extension but after the naming sequence.
        if `save_in_list` is not None, it should be a list where the images will be saved, and no file will be written.
        """
        if save_in_list is None:
            if not os.path.exists(savedir):
                os.mkdir(savedir)
        else:
            assert len(save_in_list) >= len(self._table), 'save_in_list should not be shorter than the table.'

        task_table = self._table.copy()
        # add an index column for the order in save_in_list
        task_table['index'] = np.arange(len(task_table))

        with Bar(len(task_table)) as bar:
            try_i = 0
            for try_i in range(try_loops):
                num_left = len(task_table)
                if num_left == 0:
                    break
                print(f"Try [{try_i + 1}]. {num_left} images to download...\n")
                failed_table = Table(names=task_table.colnames, dtype=task_table.dtype)
                invalid_table = Table(names=task_table.colnames, dtype=task_table.dtype)
                # the failed table will be used as the new task table
                task_table, invalid_table = self._get_figs_once(
                    task_table, failed_table, invalid_table,
                    savedir=savedir, naming_seq=naming_seq, suffix=suffix,
                    save_in_list=save_in_list,
                    overwrite=overwrite, timeout=timeout,
                    bar=bar,
                )

        print(f'Done. {len(failed_table)} items failed, {len(invalid_table)} items invalid.')
        return failed_table, invalid_table


# %% class SDSS
class SDSS(_SurveyBase):
    def __init__(self, *args, **kwargs):
        _SDSS_RENAME_DICT = {
            'plate': ['plate', 'plateid', 'plate_id', 'PLATE', 'PLATEID', 'PLATE_ID'],
            'mjd': ['mjd', 'MJD'],
            'fiberid': ['fiber', 'fiberid', 'fiber_id', 'FIBER', 'FIBERID', 'FIBER_ID'],
            'specobjid': ['specobjid', 'SPECOBJID', 'specobj_id', 'SPECOBJ_ID',
                          'spec_id', 'SPEC_ID', 'specid', 'SPECID'],
        }
        _SurveyBase.__init__(self, *args, rename_dict=_SDSS_RENAME_DICT, **kwargs)

    _timeout = 15
    name_plate_mjd_fiberid = Naming(['plate', 'mjd', 'fiberid'], '{0}, {1}, {2}')


class SDSSImg(SDSS):

    _required_cols = ['ra', 'dec']

    @classmethod
    def naming_seq(cls, table_row):
        return cls.set_naming_priority(table_row,
                                       [cls.name_plate_mjd_fiberid,
                                        cls.name_ra_dec],
                                       )

    @staticmethod
    def get_url(table_row, dr='dr17', scale=0.2, length=200, opt=''):
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

    _required_cols = ['specobjid']
    name_specobjid = Naming(['specobjid'], 'specid{0}')

    @classmethod
    def naming_seq(cls, table_row):
        return cls.set_naming_priority(table_row,
                                       [cls.name_plate_mjd_fiberid,
                                        cls.name_ra_dec,
                                        cls.name_specobjid,
                                        ],
                                       )

    @staticmethod
    def get_url(table_row, dr='dr17'):
        """
        example:
        https://skyserver.sdss.org/dr17/en/get/SpecById.ashx?id=320932083365079040
        """
        specobjid = table_row['specobjid']
        url = f"https://skyserver.sdss.org/{dr}/en/get/SpecById.ashx?id={specobjid}"
        return url


# %% class DESI
class DESIImg(_SurveyBase):

    _required_cols = ['ra', 'dec']
    _threads = 2  # DESI will send status 429 if too many requests are sent

    @classmethod
    def naming_seq(cls, table_row):
        return cls.set_naming_priority(table_row,
                                       [cls.name_ra_dec],
                                       )

    @staticmethod
    def get_url(table_row, layer='ls-dr9', bands='grz', pixscale=0.2, size=200):
        """
        example:
        """
        ra = table_row['ra']
        dec = table_row['dec']
        url = (
            "https://www.legacysurvey.org/viewer/cutout.jpg?"
            f"ra={ra}&dec={dec}&layer={layer}&pixscale={pixscale}&bands={bands}&size={size}")
        return url
