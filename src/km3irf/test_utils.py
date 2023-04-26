#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A collection of testing tools,
which can be used for writing tests.

"""

import matplotlib.pyplot as plt


# copied from gammapy tests
def mpl_plot_check():
    """Matplotlib plotting test context manager.
    It create a new figure on __enter__ and calls savefig for the
    current figure in __exit__. This will trigger a render of the
    Figure, which can sometimes raise errors if there is a problem.
    This is writing to an in-memory byte buffer, i.e. is faster
    than writing to disk.
    """
    from io import BytesIO

    class MPLPlotCheck:
        def __enter__(self):
            plt.figure()

        def __exit__(self, type, value, traceback):
            plt.savefig(BytesIO(), format="png")
            plt.close()

    return MPLPlotCheck()
