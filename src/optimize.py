#!/usr/bin/env python
"""
optimize.py

minimize an objective function using gradient descent with a simple line-search to ensure
it doesn't overshoot. there are probably scipy functions that do this, but I don't know
what the name for this algorithm would be, and anyway, I want to be able to see its
progress as it goes.
"""


from typing import Callable

import numpy as np


def minimize(func: Callable[[np.ndarray], float],
             grad: Callable[[np.ndarray], np.ndarray],
             guess: np.ndarray,
             scale: np.ndarray = None,
             tolerance: float = 1e-8,
             bounds: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
             report: Callable[[np.ndarray], None] = None):
	""" find the vector that minimizes a function of a list of points using gradient
	    descent with a dynamically chosen step size. unlike a more generic minimization
	    function, this one assumes that each datum is a vector, not a scalar, so many
	    things have one more dimension that you might otherwise expect.
	    :param func: the objective function to minimize. it takes an array of size n×2 as
	                 argument and returns a single scalar value
	    :param grad: a function that calculates the gradient of the objective function.
	                 it takes an array of size n×2 as argument and returns an array of
	                 size n×2 representing the slope of func at the given point with
	                 respect to each element of the input.
	    :param guess: the initial input to the function, from which the gradients will descend.
	    :param scale: an n-vector giving a relevant scale length for each point in the
	                  state vector. gradients will be scaled by these values. if it is not
	                  provided, we will assume that each point should move at the same
	                  speed.
	    :param tolerance: the relative tolerance. when a single step fails to reduce the
	                      error by less than this amount, the algorithm will terminate.
	    :param bounds: a list of inequality constraints on various linear combinations.
	                   each item of the list should comprise a m×n matrix that multiplies
	                   by the state array to produce a m×2 vector of tracer particle
	                   positions, a 2-vector representing the upper-left corner of the
	                   allowable bounding box, and a 2-vector representing the lower-right
	                   corner of the allowable bounding box. the algorithm will ensure
	                   that all of the tracer particles will remain inside the
	                   corresponding bounding box in the final solution.
	    :param report: a function to call on each new iteration, to provide information
	                   to the outer program while the routine is running. if noting is
	                   provided, then no reporting will be done.
	    :return: the optimal n×2 array of points
	"""
	n = guess.shape[0]
	d = guess.shape[1]
	if scale is None:
		scale = np.ones(n)
	if bounds is None:
		bounds = []
	if report is None:
		report = lambda x: None
	return guess
