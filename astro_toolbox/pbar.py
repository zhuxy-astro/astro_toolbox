#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
I try to unify the usage of progress bar from alive_bar and astropy, and also my
own progress bar.
Using the astropy progress bar will partly mess up the output in multiprocessing.
Using my own progress bar may be slow for large loops, and may not work in
Jupyter notebook.

Both of the two need additional parameters to work properly in Jupyter notebook:
    - alive_bar: force_tty=True
    - astropy: ipython_widget=True
I have redefined the two progress bars as Bar in this file, which should
automatically detect the environment. It should work fine without additional
parameters.

Usage:

from pbar import Bar
LENGTH = 100
with Bar(LENGTH) as bar:
    for i in range(LENGTH):
        bar()
"""


# %% import
import sys
from time import time

try:
    from astropy.utils.console import ProgressBar as _ProgressBar  # class
    has_astropy = True
except ImportError:
    has_astropy = False
try:
    from alive_progress import config_handler, alive_bar  # func
    config_handler.set_global(force_tty=True)
    has_alive_bar = True
except ImportError:
    has_alive_bar = False

from .misc import in_ipython


# %% redefine ProgressBar
if has_astropy:
    class ProgressBar(_ProgressBar):
        """ProgressBar aware of IPython, and accepting force_tty"""
        def __init__(self, *args, **kwargs):
            force_tty = kwargs.pop('force_tty', False)
            if 'ipython_widget' not in kwargs:
                if force_tty:
                    # when force_tty is set, follow it
                    kwargs['ipython_widget'] = True
                else:
                    # when nothing is set, automatically detect the environment
                    kwargs['ipython_widget'] = in_ipython() == 2
            super().__init__(*args, **kwargs)

        def __call__(self):
            return self.update()


# %% MyBar
class MyBar:
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


# %% Choose a progress bar
if has_alive_bar:
    Bar = alive_bar
elif has_astropy:
    Bar = ProgressBar
else:
    Bar = MyBar
