#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# %% import
from time import time
import sys
import numpy as np


# %% func: progress_bar
# def progress_bar(prog, scale, start_time):
def progress_bar(prog, scale):
    # 做一个手动的进度条，用于循环。
    # 适合for prog in range(scale)的环境
    # 但是print的过程对超大循环来说可能有点费时间。现在可用alive_bar替换。
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


# %% func: select_row
def select_id(table, value, name=['plate', 'mjd', 'fiber']):
    """
    can deal with one or multiple values and names, and multiple rows.
    returns a bool array
    When name has three attrs like ['p', 'm', 'f'],
        value is like [280, 51612, 97] or like [[280, 51612, 97], [996, 52641, 513]]
    When name has one attr like 'plate', or ['plate']
        value is like 280, or like [280, 996]
    """
    is_single_name = type(name) is str
    if not is_single_name and (len(name) == 1):
        is_single_name = True
        name = name[0]
    if is_single_name:
        try:
            len(value)
            # multiple value
            select_result = np.zeros(len(table), dtype=bool)
            for value_j in value:
                select_result |= (table[name] == value_j)
            return select_result
        except TypeError:
            # single value
            return table[name] == value

    # multiple name
    if len(value) == len(name):
        # single value
        value = [value]
    select_result = np.zeros(len(table), dtype=bool)
    for value_j in value:
        select_single = np.ones(len(table), dtype=bool)
        for name_i, value_ij in zip(name, value_j):
            select_single &= (table[name_i] == value_ij)
        select_result |= select_single
    return select_result
