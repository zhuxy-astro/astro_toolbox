#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# %% import
import sys


# %% progress bar
# unify the usage of progress bar
# using the astropy progress bar will partly mess up the output in multiprocessing
try:
    from alive_progress import alive_bar as Bar  # func
except ImportError:
    from astropy.utils.console import ProgressBar as Bar_  # class

    class Bar(Bar_):
        # def __init__(self, *args, **kwargs):
        # self.bar = Bar_(*args, **kwargs)
        def __call__(self):
            return self.update()


# def progress_bar(prog, scale, start_time):
def progress_bar(prog, scale):
    # 做一个手动的进度条，用于循环。
    # 适合for prog in range(scale)的环境
    # 但是print的过程对超大循环来说可能有点费时间。现在可用alive_bar替换。
    from time import time
    if prog == 0:
        global start_time
        start_time = time()
    prog100 = int(prog / (scale - 1) * 100)
    a = "▋" * (prog100 // 2)
    b = " " * (50 - prog100 // 2)
    dur = time() - start_time
    print("\r{:^3.0f}%[{}{}]{:.2f}s".format(prog100, a, b, dur), end="")
    sys.stdout.flush()
    if prog == scale - 1:
        print()
