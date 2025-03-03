#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# %% import
import sys
from time import time


# %% progress bar
# unify the usage of progress bar
# using the astropy progress bar will partly mess up the output in multiprocessing
"""Usage:
from misc import Bar
LENGTH = 100
with Bar(LENGTH) as bar:
    for i in range(LENGTH):
        bar()

In Jupyter notebook, add `force_tty=True` if using alive_bar
or `ipython_widget=True` if using astropy progress bar in the definition of Bar.
"""
from astropy.utils.console import ProgressBar  # class


try:
    from alive_progress import alive_bar as Bar  # func
except ImportError:
    class Bar(ProgressBar):
        def __call__(self):
            return self.update()


# %% MyBar
class MyBar:
    # 自己做的进度条。print的过程对超大循环来说可能有点费时间。现在可用alive_bar替换。
    def __init__(self, scale):
        self.scale = scale
        self.prog = 0
        self.start_time = time()

    def __enter__(self):
        return self

    def __call__(self):
        self.prog += 1
        prog100 = int(self.prog / self.scale * 100)
        a = "▋" * (prog100 // 2)
        b = " " * (50 - prog100 // 2)
        dur = time() - self.start_time
        print(f"\r{prog100:^3.0f}% ({self.prog}/{self.scale})[{a}{b}] {dur:.2f}s", end="")
        sys.stdout.flush()

    def update(self):
        self()

    def __exit__(self, exc_type, exc_val, exc_tb):
        print()
