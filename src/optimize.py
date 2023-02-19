#!/usr/bin/env python
"""
optimize.py

minimize an objective function using gradient descent with a simple line-search to ensure
it doesn't overshoot. there are probably scipy functions that do this, but I don't know
what the name for this algorithm would be, and anyway, I want to be able to see its
progress as it goes.
"""
from __future__ import annotations

import logging
from math import inf, isfinite, isnan, sqrt
from typing import Callable, Optional

import numpy as np
from numpy.typing import NDArray

from autodiff import Variable
from sparse import SparseNDArray

STEP_REDUCTION = 5.
STEP_RELAXATION = STEP_REDUCTION**1.618
LINE_SEARCH_STRICTNESS = (STEP_REDUCTION - 1)/(STEP_REDUCTION**2 - 1)
BARRIER_REDUCTION = 2.

FINE = 11
INFO = logging.INFO
logging.addLevelName(FINE, "FINE")  # define my own debug level so I don't have to see Matplotlib's debug messages

np.seterr(under="ignore", over="raise", divide="raise", invalid="raise")


class MaxIterationsException(Exception):
	pass

class ConcaveObjectiveFunctionException(Exception):
	pass


def minimize(func: Callable[[NDArray[float] | Variable], float | Variable],
             guess: NDArray[float],
             gradient_tolerance: float,
             report: Optional[Callable[[NDArray[float], float, NDArray[float], NDArray[float]], None]] = None,
             ) -> MinimizationResult:
	""" find the vector that minimizes a function of a list of points using a twoth-order gradient-descent-type-thing
	    with a dynamically chosen step size. unlike a more generic minimization routine, this one assumes that each
	    datum is a vector, not a scalar, so many things have one more dimension than you might otherwise expect.
	    :param func: the objective function to minimize. it takes an array of size m×n as argument and returns a single
	                 scalar value. if at any point it returns -inf, whatever state produced it will be immediately
	                 returned. points that return +inf will, naturally, be avoided at all costs.
	    :param guess: the initial input to the function, from which the gradients will descend.
	    :param gradient_tolerance: the absolute tolerance. if the magnitude of the gradient dips below this at any given
	                               point, we are done.
	    :param report: an optional function that will be called each time a line search is completed, to provide real-
	                   time information on how the fitting routine is going. it takes as arguments the current state,
	                   the current value of the function, the current gradient magnitude, the previous step, and the
	                   fraction of the step that is currently getting projected away by the bounds.
	    :return: the optimal m×n array of points
	"""
	# if no report function is provided, default it to a null callable
	if report is None:
		def report(*_):
			pass

	# redefine the objective function to have some checks bilt in
	def get_value(x: np.ndarray) -> float:
		value = func(x)
		if isnan(value):
			raise RuntimeError(f"there are nan values at x = {x}")
		return value
	# define a utility function to use Variable to get the gradient of the value
	def get_gradient(x: NDArray[float]) -> tuple[NDArray[float], SparseNDArray]:
		variable = func(Variable.create_independent(x))
		if np.any(np.isnan(variable.gradient)):
			raise RuntimeError(f"there are nan gradients at x = {x}")
		return variable.gradient, variable.hessian

	initial_value = get_value(guess)

	# silently immediately terminate if we’re already at -inf
	if initial_value == -inf:
		return MinimizationResult("feasible", guess, initial_value)
	# or complain if the input is not feasible
	elif not isfinite(initial_value):
		raise RuntimeError(f"the objective function returned an invalid initial value: {initial_value}")

	# calculate the initial gradient
	gradient, hessian = get_gradient(guess)
	if gradient.shape != guess.shape:
		raise ValueError(f"the gradient function returned the wrong shape ({gradient.shape}, should be {guess.shape})")
	identity = SparseNDArray.identity(hessian.dense_shape)

	# instantiate the loop state variables
	value = initial_value
	state = guess
	step_limiter = np.quantile(abs(hessian.diagonal()), 1e-2)
	# descend until we can't descend any further
	num_line_searches = 0
	while True:
		# do a line search to choose a good step size
		num_step_sizes = 0
		while True:
			# choose a step by minimizing this quadratic approximation, projecting onto the legal subspace
			if step_limiter < abs(hessian).max()*10:
				step = -reshape_inverse_matmul(hessian + identity*step_limiter, gradient)
			else:  # if the step limiter is big enuff, use this simpler approximation
				step = -gradient/step_limiter
			new_state = state + step
			new_value = get_value(new_state)
			# if this is infinitely good, just return it without further question
			if new_value == -inf:
				logging.log(FINE, f"Reached the valid domain in {num_line_searches} iterations.")
				return MinimizationResult("feasible", new_state, new_value)
			# if the line search condition is met, take it
			if new_value < value + LINE_SEARCH_STRICTNESS*np.sum(step*gradient):
				logging.log(FINE, f"{step_limiter:.2g} -> !! good !! (stepd {np.linalg.norm(step):.3g})")
				break
			elif new_value < value:
				logging.log(FINE, f"{step_limiter:7.2g} -> .. not better enuff")
			else:
				logging.log(FINE, f"{step_limiter:7.2g} -> .. not better ({value:.12g} -> {new_value:.12g})")

			# if the condition is not met, decrement the step size and try agen
			step_limiter *= STEP_REDUCTION
			num_step_sizes += 1
			# keep track of the number of step sizes we've tried
			if num_step_sizes > 100:
				raise RuntimeError("line search did not converge")

		# take the new state and error value
		state = new_state
		value = new_value
		# recompute the gradient once per outer loop
		gradient, hessian = get_gradient(state)

		# if the termination condition is met, finish
		gradient_magnitude = np.linalg.norm(gradient)
		report(state, value, gradient, step)
		# if the termination condition is met, finish
		if gradient_magnitude < gradient_tolerance:
			logging.log(INFO, f"Completed in {num_line_searches} iterations.")
			return MinimizationResult("optimal", state, value)

		# set the step size back a bit
		step_limiter /= STEP_RELAXATION
		# keep track of the number of iterations
		num_line_searches += 1
		if num_line_searches >= 1e5:
			raise RuntimeError(f"algorithm did not converge in {num_step_sizes} iterations")


def minimize_with_bounds(objective_func: Callable[[NDArray[float] | Variable], float | Variable],
                         guess: NDArray[float],
                         gradient_tolerance: float,
                         barrier_tolerance: float,
                         bounds_matrix: Optional[SparseNDArray] = None,
                         bounds_limits: Optional[NDArray[float]] = None,
                         report: Optional[Callable[[NDArray[float], float, NDArray[float], NDArray[float]], None]] = None,
                         ) -> MinimizationResult:
	""" find the vector that minimizes a function of a list of points
	        argmin_x(f(x))
	    subject to a convex constraint expressed as
	        all(bounds_matrix@x <= bounds_limits).
	    using an interior-point method.  each step will be solved using a twoth-order gradient-descent-type-thing with
	    a dynamically chosen step size.
	    :param objective_func: the objective function to minimize. it takes an array of size m×n as argument and returns
	                           a single scalar value. if at any point it returns -inf, whatever state produced it will
	                           be immediately returned. points that return +inf will be avoided at all costs, naturally.
	    :param guess: the initial input to the function, from which the gradients will descend.
	    :param gradient_tolerance: the gradient descent absolute tolerance. when the magnitude of the gradient dips
	                               below this during the inner iteration, we reduce the barrier parameter.
	    :param barrier_tolerance: the interior point absolute tolerance. when we estimate that the barrier function is
	                              displacing the result by less than this amount, we are done.
	    :param bounds_matrix: a list of inequality constraints on various linear combinations. it should be some object
	                          that matmuls by the state array to produce an l×n vector of tracer particle positions.
	    :param bounds_limits: the values of the inequality constraints. should be something that can broadcast to l×n,
	                          representing the maximum coordinates of each tracer particle.
	    :param report: an optional function that will be called each time a line search is completed, to provide real-
	                   time information on how the fitting routine is going. it takes as arguments the current state,
	                   the current value of the function, the current gradient magnitude, the previous step, and the
	                   fraction of the step that is currently getting projected away by the bounds.
	    :return: the optimal m×n array of points
	"""
	# first of all, skip the bounding if at all possible
	if np.all((bounds_limits > 0) & np.isinf(bounds_limits)):
		return minimize(objective_func, guess, gradient_tolerance, report)
	# also, check that we’re not on the bound, because that will cause problems
	elif np.any(bounds_matrix@guess >= bounds_limits):
		raise ValueError("the initial guess must have some clearance from the feasible space.")

	# set up the barrier function
	def barrier_func(x):
		if np.any(bounds_matrix@x >= bounds_limits):
			return inf
		else:
			return -(np.log(-(bounds_matrix@x - bounds_limits))).sum()
	# so that you can set the initial barrier parameter
	guess_variable = Variable.create_independent(guess)
	initial_objective_value = objective_func(guess_variable)
	if initial_objective_value == -inf:
		return MinimizationResult("feasible", guess, initial_objective_value)
	elif type(initial_objective_value) is not Variable or not isfinite(initial_objective_value.value):
		raise RuntimeError(f"the objective function returned an invalid initial value: {initial_objective_value}")
	else:
		initial_objective_force = abs(initial_objective_value.gradient)
	initial_barrier_force = abs(barrier_func(guess_variable).gradient)
	near = initial_barrier_force > np.max(initial_barrier_force)/1.1
	barrier_height = 2*np.max(initial_objective_force[near]/initial_barrier_force[near])

	# set up the new objective function (the old one plus the barrier function)
	def compound_func(x):
		objective_value = objective_func(x)
		barrier_value = barrier_func(x)
		if barrier_value == inf:  # important note: if both values are opposite infs, the barrier’s +inf should win
			return inf
		else:
			return objective_value + barrier_height*barrier_value

	# finally, do the minimization, gradually decreasing the barrier height
	state = guess
	num_iterations = 0
	while True:
		result = minimize(compound_func, state, gradient_tolerance, report)
		if result.reason == "feasible":
			return result
		elif np.max(abs(result.state - state)) < barrier_tolerance*(1 - 1/BARRIER_REDUCTION):
			logging.log(INFO, f"Completed interior point method in {num_iterations} steps.")
			return result
		elif num_iterations >= 10_000:
			raise RuntimeError("Interior point method did not converge.")
		logging.log(FINE, "reached temporary solution; reducing barrier parameter.")
		barrier_height /= BARRIER_REDUCTION
		state = result.state
		num_iterations += 1


def polytope_project(point: NDArray[float],
                     polytope_mat: SparseNDArray, polytope_lim: NDArray[float]
                     ) -> NDArray[float]:
	""" project a given point onto a polytope defined by the inequality
	        all(polytope_mat@point <= polytope_lim + tolerance)
	    :param point: the point to project
	    :param polytope_mat: the normal vectors of the polytope faces, by which the polytope is defined
	    :param polytope_lim: the quantity that defines the size of the polytope.  it may be an array
	                         if point is 2d.  a point is in the polytope iff
	                         polytope_mat @ point[:, k] <= polytope_lim[k] for all k
	"""
	if point.ndim == 2:
		if point.shape[1] != polytope_lim.shape[1]:
			raise ValueError(f"if you want this fancy functionality, the shapes must match")
		# if there are multiple dimensions, do each dimension one at a time; it's faster, I think
		return np.stack([polytope_project(point[:, k], polytope_mat, polytope_lim[:, k])
		                 for k in range(point.shape[1])]).T

	def distance_from_point(x):
		return ((x - point)**2).sum()
	guess = crudely_polytope_project(point, polytope_mat, polytope_lim)
	crude_distance = sqrt(distance_from_point(guess - point))
	return minimize_with_bounds(distance_from_point, guess,
	                            gradient_tolerance=1e-3*crude_distance,
	                            barrier_tolerance=1e-3*crude_distance,
	                            bounds_matrix=polytope_mat,
	                            bounds_limits=polytope_lim).state


def crudely_polytope_project(point: NDArray[float],
                             polytope_mat: SparseNDArray, polytope_lim: NDArray[float]
                             ) -> NDArray[float]:
	""" find a point near the given point that is inside the specified polytope
	    :param point: the point to be projected
	    :param polytope_mat: the normal vectors of the polytope faces, by which the polytope is defined
	    :param polytope_lim: the quantity that defines the size of the polytope on each face
	    :return: a point such that all(polytope_mat@point <= polytope_lim)
	"""
	if point.ndim == 2:
		if point.shape[1] != polytope_lim.shape[1]:
			raise ValueError(f"if you want this fancy functionality, the shapes must match")
		# if there are multiple dimensions, do each dimension one at a time
		return np.stack([crudely_polytope_project(point[:, k], polytope_mat, polytope_lim[:, k])
		                 for k in range(point.shape[1])]).T
	polytope_magnitudes = (polytope_mat**2).sum(axis=[1])
	residuals = polytope_mat@point - polytope_lim
	steps = polytope_mat*(residuals/polytope_magnitudes)[..., np.newaxis]
	num_iterations = 0
	while True:
		if np.all(residuals <= 0):
			return point
		for i in range(polytope_mat.shape[0]):
			if residuals[i] > 0:
				point = point - np.array(steps[i, :])*1.2  # this 1.2 makes it so it reaches a solution in finite iterations
				residuals = polytope_mat@point - polytope_lim
				steps = polytope_mat*(residuals/polytope_magnitudes)[:, np.newaxis]
		num_iterations += 1
		if num_iterations > 20:
			raise MaxIterationsException("this crude polytope projection really autn't take more that 2 iterations")


def reshape_inverse_matmul(A: SparseNDArray, b: NDArray[float]) -> NDArray[float]:
	""" a call to SparseNDArray.inverse_matmul() that reshapes things if they’re too dimensional """
	return A.reshape((b.size, b.size), 1).inverse_matmul(b.ravel()).reshape(b.shape)


class MinimizationResult:
	def __init__(self, reason: str, state: NDArray[float], objective: float):
		""" :param reason: either "optimal" if it found a local optimum or "feasible" if the objective function went to
		                   -inf. if it’s anything else, it’ll just raise a RuntimeError.
		    :param state: the vector that produces the minimized value
		    :param objective: the minimized value of the objective function
		"""
		self.reason = reason
		self.state = state
		self.objective = objective


def test():
	import matplotlib.pyplot as plt
	import numpy as np

	polytope_matrix = SparseNDArray.from_coordinates(
		[2],
		np.array([[[0], [1]], [[0], [1]], [[0], [1]], [[0], [1]], [[0], [1]]]),
		np.array([[.7, .3], [0., 1.1], [0., -.8], [-.6, 0.], [-.7, -.7]]))
	polytope_limits = np.array(1.)

	X, Y = np.meshgrid(np.linspace(-2, 2, 101), np.linspace(-2, 2, 101), indexing="ij")

	point = np.array([-2., -1.6])
	projection = polytope_project(point, polytope_matrix, polytope_limits)
	polytope_positions = np.array(polytope_matrix)[:, 0, None, None]*X[None, :, :] + \
	                     np.array(polytope_matrix)[:, 1, None, None]*Y[None, :, :]
	polytope_inness = np.max(polytope_positions, axis=0)

	plt.contour(X, Y, polytope_inness, levels=[polytope_limits], colors="k")
	plt.plot(point[0], point[1], "x")
	plt.plot(projection[0], projection[1], "o")
	print(projection)
	plt.axis("equal")
	plt.show()

	x0 = np.array([1., -1.])
	gradient = np.array([2.5, -3.0])
	hessian = SparseNDArray.from_coordinates([2],
	                                         np.array([[[0], [1]], [[0], [1]]]),
	                                         np.array([[1.0, -0.9], [-0.9, 1.0]]))
	I = SparseNDArray.identity(hessian.dense_shape)

	solutions = []
	for caution in [0, .01, .1, 1, 10, 100, 10000]:
		print(f"using caution of {caution}")
		def objective(state):
			Δx = state - x0
			H = hessian + I*caution
			return 1/2*(Δx*(H@Δx)).sum() + (Δx*gradient).sum()
		solution = minimize_with_bounds(objective, x0,
		                                bounds_matrix=polytope_matrix,
		                                bounds_limits=polytope_limits,
		                                gradient_tolerance=1e-6,
		                                barrier_tolerance=1e-6).state
		print("done!\n\n")
		solutions.append(solution)

	plt.contour(X, Y, polytope_inness, levels=[polytope_limits], colors="k")
	dX, dY = X - x0[0], Y - x0[1]
	plt.contourf(X, Y, dX*gradient[0] + dY*gradient[1] +
	             1/2*(dX**2*np.array(hessian)[0, 0] + 2*dX*dY*np.array(hessian)[0, 1] + dY**2*np.array(hessian)[1, 1]))
	plt.axis("equal")

	plt.plot(x0[0], x0[1], "wo")
	plt.plot([p[0] for p in solutions], [p[1] for p in solutions], "w-x")
	plt.show()


if __name__ == "__main__":
	test()
