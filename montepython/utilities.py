# -*- coding: utf-8 -*-
"""
Utility functions.
"""

from __future__ import (absolute_import, division, print_function, 
                        unicode_literals)

import numpy as np

def ball(p0, sigma, size=1):
    """
    Create a Gaussian ball around a point.

    Parameters
    ----------

    p0 : list
        Position vector to create ball around.
    sigma : float
        Standard deviation of Gaussian.
    size : int
        Number of balls to create.
    """
    return [p0 + sigma * np.random.normal(size=len(p0))
            for i in range(size)]