from . import bar
Bar = bar.Bar


def in_ipython():
    """Return value:
    0: Standard Python interpreter
    1: IPython terminal
    2: Jupyter notebook or qtconsole
    3: Other type (maybe unknown)
    """
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            # Jupyter notebook or qtconsole
            return 2
        elif shell == "TerminalInteractiveShell":
            # Terminal running IPython
            return 1
        else:
            # Other type (maybe unknown)
            return -1
    except NameError:
        # get_ipython not defined, so likely standard Python interpreter
        return 0


is_in_ipython = in_ipython()
