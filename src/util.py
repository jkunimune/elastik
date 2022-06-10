#!/usr/bin/env python
"""
util.py

some handy utility functions that are used in multiple places
"""

import numpy as np


def bin_centers(bin_edges: np.ndarray) -> np.ndarray:
	""" calculate the center of each bin """
	return (bin_edges[1:] + bin_edges[:-1])/2

def bin_index(x: float or np.ndarray, bin_edges: np.ndarray) -> float or np.ndarray:
	""" I dislike the way numpy defines this function
	    :param: coerce whether to force everything into the bins (useful when there's roundoff)
	"""
	return np.where(x < bin_edges[-1], np.digitize(x, bin_edges) - 1, bin_edges.size - 2)

def wrap_angle(x: float or np.ndarray) -> float or np.ndarray: # TODO: come up with a better name
	""" wrap an angular value into the range [-np.pi, np.pi) """
	return x - np.floor((x + np.pi)/(2*np.pi))*2*np.pi
