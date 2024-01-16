#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# %% import
import requests
import os
from astropy.table import Table
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


# %% class _SurveyBase
class _SurveyBase:
    """Base class for surveys.
    The setting of table is not necessary when using show_fig or get_url.
    The ra and dec is always required, and should be in degrees.
    """
    def _rename_col(self, possible_names, new_name):
        """convert possible variations of column names to a standard name
        OK if there is no match.
        """
        for name in possible_names:
            if name in self._table.colnames:
                self._table.rename_column(name, new_name)
                return

    def __init__(self, table=None):
        """table could be an astropy table or a row
        """
        self._TIMEOUT = 30  # seconds. Different surveys may need different timeout
        self._THREADS = 50  # number of threads for downloading images
        self._FILE_SUFFIX = '.jpg'

        if table is None:
            self._table = None
            return
        # with table available
        if isinstance(table, Table):
            self._table = table.copy()  # copying is necessary to avoid changing the original table
        else:
            self._table = Table(table)

        # converting ra and dec
        self._rename_col(['ra', 'RA'], 'ra')
        self._rename_col(['dec', 'DEC', 'Dec'], 'dec')
        # assert set(['ra', 'dec']).issubset(self._table.colnames), \
        # 'table should include ra and dec'

    def _short_ra_dec(self, ra_or_dec):
        return f'{ra_or_dec:.4f}'

    def _get_name_ra_dec(self, table_row):
        """default name based on ra and dec
        """
        name = (f'RA{self._short_ra_dec(table_row["ra"])},'
                f'Dec{self._short_ra_dec(table_row["dec"])}')
        return name

    _get_name = _get_name_ra_dec

    def get_url(self, table_row, *args, **kwargs):
        """always different for different surveys
        """
        raise NotImplementedError

    def _get_response(self, table_row):
        url = self.get_url(table_row)
        response = requests.get(url, stream=True, timeout=self._TIMEOUT)
        return response

    def show_fig(self, table_row, timeout=None):
        """show the image in plt.
        Setting of table is not necessary when using this function.
        """
        if timeout is not None:
            self._TIMEOUT = timeout

        response = self._get_response(table_row)
        fig = Image.open(response.raw)

        fig = plt.figure(figsize=(6.4, 4.8))
        ax = fig.add_subplot(111)
        ax.set_title(self._get_name(table_row),
                     fontfamily='sans-serif', fontsize=16)
        ax.imshow(fig)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.show()

    def _get_one_fig(
        self, table_row, failed_table,
        savedir, filename,
        overwrite, bar,
    ):
        """download and write one image
        """
        if filename is None:
            filename = self._get_name(table_row) + self._FILE_SUFFIX
        savepath = os.path.join(savedir, filename)
        if not overwrite and os.path.exists(savepath):
            update(bar)
            return

        # putting try inside this function can avoid error raising in multitasking
        try:
            response = self._get_response(table_row)
            # TODO: when to raise?
            response.raise_for_status()
            fig = response.content
            with open(savepath, 'wb') as f:
                f.write(fig)
            update(bar)
        except Exception:
            failed_table.add_row(table_row)

    def _get_figs_once(self, task_table, failed_table, **kwargs):
        """download and write multiple images using multiple threads
        everything in kwargs will be passed to _get_one_fig
        """
        pool = ThreadPool(self._THREADS)
        func = partial(self._get_one_fig, failed_table=failed_table, **kwargs)
        pool.map(func, task_table)
        pool.close()
        pool.join()
        return failed_table

    def get_figs(self, savedir='img', overwrite=False, filename=None, try_loops=3, timeout=None):
        """filename is set to Survey._get_name(table_row) + '.jpg' by default
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
                # the failed table will be used as the new task table
                task_table = self._get_figs_once(
                    task_table, failed_table,
                    savedir=savedir, filename=filename,
                    overwrite=overwrite,
                    bar=bar,
                )
                # task_table = failed_table.copy()

        print(f'Done. {len(failed_table)} items failed.')
        return failed_table


# %% class SDSS
class SDSS(_SurveyBase):
    def _get_sdss_name(self, table_row):
        name = f"{table_row['plate']},{table_row['mjd']},{table_row['fiber']}"
        return name

    def __init__(self, *args, **kwargs):
        _SurveyBase.__init__(self, *args, **kwargs)
        self._TIMEOUT = 15
        if self._table is None:
            return
        self._rename_col(['plate', 'plateid', 'plate_id', 'PLATE', 'PLATEID', 'PLATE_ID'], 'plate')
        self._rename_col(['mjd', 'MJD'], 'mjd')
        self._rename_col(['fiber', 'fiberid', 'fiber_id', 'FIBER', 'FIBERID', 'FIBER_ID'], 'fiber')
        self._rename_col(['specobjid', 'SPECOBJID', 'specobj_id', 'SPECOBJ_ID', 'spec_id', 'SPEC_ID', 'specid', 'SPECID'],
                         'specobjid')


class SDSSImg(SDSS):
    if set(['plate', 'mjd', 'fiber']).issubset(self._table.colnames):
        self._get_name = self._get_sdss_name

    def get_url(self, table_row, scale=0.2, length=200, opt=''):
        """
        `opt` sets the mark on the image. 'G' for galaxy with crosses, 'S' for star with a red square.
        example:
        https://skyserver.sdss.org/dr17/SkyServerWS/ImgCutout/getjpeg?TaskName=Skyserver.Explore.Image&ra=195.299786348959%20&dec=29.6641012044628&scale=0.2&width=200&height=200&opt=G
        """
        ra = table_row['ra']
        dec = table_row['dec']
        url = (
            "https://skyserver.sdss.org/dr17/SkyServerWS/ImgCutout/getjpeg?TaskName=Skyserver.Explore.Image&"
            f"ra={ra}%20&dec={dec}&scale={scale}&width={length}&height={length}&opt={opt}")
        return url


class SDSSSpec(SDSS):
    def __init__(self, *args, **kwargs):
        SDSS.__init__(self, *args, **kwargs)
        assert 'specobjid' in self._table.colnames, \
            'table should include specobjid'

    # TODO: if no matching? if nan value? save filename?
    # TODO: what is it like in plt?
    def _get_name(self, table_row):
        return self._get_name(table_row) + 'spec'

    def get_url(self, table_row):
        """
        example:
        https://skyserver.sdss.org/dr17/en/get/SpecById.ashx?id=320932083365079040
        """
        specobjid = table_row['specobjid']
        url = f"https://skyserver.sdss.org/dr17/en/get/SpecById.ashx?id={specobjid}"
        return url

# %% class DESI
# TODO: if matching failed, what is the response in requests?
# TODO: find a scale.
