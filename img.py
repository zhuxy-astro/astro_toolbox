#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# %% import
import requests
import os
from astropy.table import Table
from PIL import Image
import matplotlib.pyplot as plt
import threading
from queue import Queue

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


# %% class Survey
class Survey:
    """Base class for surveys.
    The setting of table is not necessary when using show_img or get_url.
    The ra and dec is always required, and should be in degrees.
    """
    def _rename_col(self, possible_names, new_name):
        """convert possible variations of column names to a standard name
        """
        for name in possible_names:
            if name in self._table.colnames:
                self._table.rename_column(name, new_name)
                return

    def __init__(self, table=None):
        """table could be an astropy table or a row
        """
        self.timeout = 30  # seconds. Different surveys may need different timeout

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
        assert set(['ra', 'dec']).issubset(self._table.colnames), \
            'table should include ra and dec'

    def _short_ra_dec(self, ra_or_dec):
        return f'{ra_or_dec:.4f}'

    def _get_name(self, table_row):
        """default name based on ra and dec
        """
        name = (f'RA{self._short_ra_dec(table_row["ra"])},'
                f'Dec{self._short_ra_dec(table_row["dec"])}')
        return name

    def get_url(self, table_row, *args, **kwargs):
        """always different for different surveys
        """
        raise NotImplementedError

    def _get_response(self, table_row):
        url = self.get_url(table_row)
        response = requests.get(url, stream=True, timeout=self.timeout)
        return response

    def show_img(self, table_row):
        """show the image in plt.
        Setting of table is not necessary when using this function.
        """
        response = self._get_response(table_row)
        img = Image.open(response.raw)

        fig = plt.figure(figsize=(6.4, 4.8))
        ax = fig.add_subplot(111)
        ax.set_title(self._get_name(table_row),
                     fontfamily='sans-serif', fontsize=16)
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.show()

    def _get_one_img(
        self,
        table_row, failed_queue,
        savedir, filename,
        overwrite, bar,
    ):
        """download and write one image
        """
        if filename is None:
            filename = self._get_name(table_row) + '.jpg'
        savepath = os.path.join(savedir, filename)
        if not overwrite and os.path.exists(savepath):
            update(bar)
            return

        # putting try inside this function can avoid error raising in multitasking
        try:
            response = self._get_response(table_row)
            img = response.content
            with open(savepath, 'wb') as f:
                f.write(img)
            update(bar)
        except Exception:
            failed_queue.put(table_row)

    def _get_imgs_once(self, task_queue, **kwargs):
        """download and write multiple images using multiple threads
        everything in kwargs will be passed to _get_one_img
        """
        threads = []
        failed_queue = Queue()
        while not task_queue.empty():
            table_row = task_queue.get()
            download_thread = threading.Thread(
                target=self._get_one_img,
                args=(table_row, failed_queue),
                kwargs=kwargs
            )

            download_thread.daemon = True
            download_thread.start()
            threads.append(download_thread)
            task_queue.task_done()
        # wait for all threads to finish
        task_queue.join()
        for t in threads:
            t.join()
        return failed_queue

    def get_imgs(self, savedir='img', overwrite=False, filename=None, try_loops=3):
        """filename is set to Survey._get_name(table_row) + '.jpg' by default
        """
        if not os.path.exists(savedir):
            os.mkdir(savedir)

        task_queue = Queue()
        for table_row in self._table:
            task_queue.put(table_row)

        with progressbar(task_queue.qsize()) as bar:
            try_i = 0
            while try_i < try_loops and task_queue.qsize() > 0:
                try_i += 1
                print(f"Try {try_i}. {task_queue.qsize()} images to download ...\n")
                # the failed queue will be used as the new task queue
                task_queue = self._get_imgs_once(
                    task_queue,
                    savedir=savedir, filename=filename,
                    overwrite=overwrite,
                    bar=bar,
                )

        # save the failed table
        failed_table = Table(names=self._table.colnames, dtype=self._table.dtype)
        for table_row in task_queue.queue:
            failed_table.add_row(table_row)

        print(f'Done. {len(failed_table)} items failed.')
        return failed_table


# %% class SDSS
class SDSS(Survey):
    def _get_sdss_name(self, table_row):
        name = f"{table_row['plate']},{table_row['mjd']},{table_row['fiber']}"
        return name

    def __init__(self, *args, **kwargs):
        Survey.__init__(self, *args, **kwargs)
        self.timeout = 15
        if self._table is None:
            return
        self._rename_col(['plate', 'plateid', 'plate_id', 'PLATE', 'PLATEID', 'PLATE_ID'], 'plate')
        self._rename_col(['mjd', 'MJD'], 'mjd')
        self._rename_col(['fiber', 'fiberid', 'fiber_id', 'FIBER', 'FIBERID', 'FIBER_ID'], 'fiber')
        if set(['plate', 'mjd', 'fiber']).issubset(self._table.colnames):
            self._get_name = self._get_sdss_name

    def get_url(self, table_row, scale=0.2, length=200, opt=''):
        """ra, dec are in degrees
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

# %% class DESI
# TODO : if matching failed? Save into jpg.
