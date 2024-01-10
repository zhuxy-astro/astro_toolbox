#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# %% import
import requests
import os
from astropy.table import Table
from astropy.utils.console import ProgressBar
from PIL import Image
import matplotlib.pyplot as plt
# import multiprocessing as mp
import multitasking
import signal
# from retry import retry
signal.signal(signal.SIGINT, multitasking.killall)

# %% func: get image
def get_sdss_url(ra, dec, scale=0.2, length=200, opt=''):
    """ra, dec are in degrees
    example:
    https://skyserver.sdss.org/dr17/SkyServerWS/ImgCutout/getjpeg?TaskName=Skyserver.Explore.Image&ra=195.299786348959%20&dec=29.6641012044628&scale=0.2&width=200&height=200&opt=G
    """
    url = (
        "https://skyserver.sdss.org/dr17/SkyServerWS/ImgCutout/getjpeg?TaskName=Skyserver.Explore.Image&"
        f"ra={ra}%20&dec={dec}&scale={scale}&width={length}&height={length}&opt={opt}")
    return url


def short_ra_dec(ra_or_dec):
    return f'{ra_or_dec:.4f}'


def show_sdss_img(row=None, ra_dec=None):
    try:
        ra = row['ra']
        dec = row['dec']
    except Exception:
        try:
            ra, dec = ra_dec
        except Exception:
            raise Exception('No coordinate is given.')

    url = get_sdss_url(ra, dec)
    response = requests.get(url, stream=True, timeout=20)
    img = Image.open(response.raw)

    fig = plt.figure(figsize=(6.4, 4.8))
    ax = fig.add_subplot(111)
    ax.set_title(f'RA, Dec: {short_ra_dec(ra)}, {short_ra_dec(dec)}',
                 fontfamily='sans-serif', fontsize=16)
    ax.imshow(img)
    fig.show()

@multitasking.task
def _get_one_sdss_img(
    table_i, failed_table,
    savedir, filename=None,
    bar=None,
    timeout=20,  # seconds
    **kwargs
    ):
    try:
        ra = table_i['ra']
        dec = table_i['dec']
        url = get_sdss_url(ra, dec)
        response = requests.get(url, timeout=timeout)
        img = response.content
        if filename is None:
            filename = f'RA{short_ra_dec(ra)}Dec{short_ra_dec(dec)}.jpg'
        savepath = os.path.join(savedir, filename)
        with open(savepath, 'wb') as f:
            f.write(img)
    except Exception:
        # keeping the adding here will allow the failed_table to be updated with multiprocessing
        failed_table.add_row(table_i)
    finally:
        if bar is not None:
            bar.update()


def get_sdss_imgs(table, savedir='img', **kwargs):
    """table should include plate, mjd, fiber, ra, dec
    TODO: overwrite or not?
    """
    assert set(['plate', 'mjd', 'fiber', 'ra', 'dec']).issubset(table.colnames), \
        'table should include plate, mjd, fiber, ra, dec'

    if not os.path.exists(savedir):
        os.mkdir(savedir)

    length = len(table)
    print(f'{length} images to download ...')
    failed_table = Table(names=table.colnames, dtype=table.dtype)  # empty
    with ProgressBar(length) as bar:
        for table_i in table:
            _get_one_sdss_img(
                table_i, failed_table,
                savedir=savedir,
                filename=f"{table_i['plate']},{table_i['mjd']},{table_i['fiber']}.jpg",
                bar=bar,
                **kwargs
            )
        multitasking.wait_for_tasks()
    print(f'Done. {len(failed_table)} items failed.')
    return failed_table


