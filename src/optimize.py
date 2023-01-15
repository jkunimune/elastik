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
from math import inf, isfinite, sqrt, isnan
from typing import Callable, Optional

import numpy as np
from numpy.typing import NDArray

from autodiff import Variable
from sparse import DenseSparseArray

STEP_REDUCTION = 5.
STEP_RELAXATION = STEP_REDUCTION**1.618
LINE_SEARCH_STRICTNESS = (STEP_REDUCTION - 1)/(STEP_REDUCTION**2 - 1)

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
             bounds_matrix: Optional[DenseSparseArray] = None,
             bounds_limits: Optional[NDArray[float]] = None,
             report: Optional[Callable[[NDArray[float], float, NDArray[float], NDArray[float], float], None]] = None,
             backup_func: Optional[Callable[[NDArray[float] | Variable], float | Variable]] = None,
             ) -> NDArray[float]:
	""" find the vector that minimizes a function of a list of points using a twoth-order projected-
	    gradient-descent-type-thing with a dynamically chosen step size. unlike a more generic
	    minimization routine, this one assumes that each datum is a vector, not a scalar, so many
	    things have one more dimension than you might otherwise expect.
	    :param func: the objective function to minimize. it takes an array of size n×m as
	                 argument and returns a single scalar value
	    :param guess: the initial input to the function, from which the gradients will descend.
	    :param gradient_tolerance: the absolute tolerance. if the portion of the magnitude of the
	                               gradient that is not against the bounds dips below this at any
	                               given point , we are done.
	    :param bounds_matrix: a list of inequality constraints on various linear combinations.
	                          it should be some object that matrix-multiplies by the state array to
	                          produce an l×m vector of tracer particle positions
	    :param bounds_limits: the values of the inequality constraints. should be something that can
	                          broadcast to l×m, representing the maximum coordinates of each tracer
	                          particle.
	    :param report: an optional function that will be called each time a line search is
	                   completed, to provide real-time information on how the fitting routine is
	                   going. it takes as arguments the current state, the current value of the
	                   function, the current gradient magnitude, the previous step, and the fraction
	                   of the step that is currently getting projected away by the bounds.
	    :param backup_func: an optional additional objective function to use when func is
	                        nonapplicable. specificly, when the primary objective function is only
	                        defined in a certain domain but the initial guess may be outside of it,
	                        the backup can be used to push the state vector into that domain. it
	                        should return smaller and smaller values as the state approaches the
	                        valid domain and -inf for states inside it. if a -inf in achieved with
	                        the backup function, it will immediately switch to the primary function.
	                        if -inf is never returned and the backup function converges, that
	                        minimum will be returnd.
	    :return: the optimal n×m array of points
	"""
	# start by checking the guess agenst the bounds
	if bounds_matrix is None:
		if bounds_limits is not None:
			raise ValueError("you mustn't pass bounds_limits without bounds_matrix")
		bounds_matrix = DenseSparseArray.zeros((0,), guess.shape[:1])
		bounds_limits = np.full((1, *guess.shape[1:]), inf)
	else:
		if bounds_limits is None:
			raise ValueError("you mustn't pass bounds_matrix without bounds_limits")
		guess = polytope_project(guess, bounds_matrix, bounds_limits)

	# if a backup objective function is provided, start with that
	if backup_func is not None:
		followup_func = func
		func = backup_func
	else:
		followup_func = None

	# redefine the objective function to have some checks bilt in
	def get_value(x: np.ndarray) -> float:
		value = func(x)
		if isnan(value):
			raise RuntimeError(f"there are nan values at x = {x}")
		return value
	# define a utility function to use Variable to get the gradient of the value
	def get_gradient(x: np.ndarray) -> tuple[NDArray[float], DenseSparseArray]:
		variable = func(Variable(x, independent=True))
		if np.any(np.isnan(variable.gradients)):
			raise RuntimeError(f"there are nan gradients at x = {x}")
		return np.array(variable.gradients), variable.hessians.make_axes_dense(2)

	initial_value = get_value(guess)

	# check just in case we instantly fall thru to the followup function
	if initial_value == -inf and followup_func is not None:
		func = followup_func
		followup_func = None
		initial_value = get_value(guess)
	elif not isfinite(initial_value):
		raise RuntimeError(f"the objective function returned an invalid initial value: {initial_value}")

	# calculate the initial gradient
	gradient, hessian = get_gradient(guess)
	if gradient.shape != guess.shape:
		raise ValueError(f"the gradient function returned the wrong shape ({gradient.shape}, should be {guess.shape})")
	identity = DenseSparseArray.identity(hessian.dense_shape)

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
			try:
				if step_limiter < abs(hessian).max()*10:
					new_state, ideal_new_state = minimize_quadratic_in_polytope(
						state, hessian + identity*step_limiter, gradient,
						bounds_matrix, bounds_limits,
						return_unbounded_solution=True)
				else:  # if the step limiter is big enuff, use this simpler approximation
					ideal_new_state = state - gradient/step_limiter
					new_state = polytope_project(ideal_new_state, bounds_matrix, bounds_limits)
			except ConcaveObjectiveFunctionException:
				pass
				logging.log(FINE, f"{step_limiter:7.2g} -> xx not valid")
			else:
				ideal_step = ideal_new_state - state
				actual_step = new_state - state
				new_value = get_value(new_state)
				# if this is infinitely good, jump to the followup function now
				if new_value == -inf and followup_func is not None:
					logging.log(FINE, f"Reached the valid domain in {num_line_searches} iterations.")
					return minimize(followup_func, new_state, gradient_tolerance,
					                bounds_matrix, bounds_limits, report, None)
				# if the line search condition is met, take it
				if new_value < value + LINE_SEARCH_STRICTNESS*np.sum(actual_step*gradient):
					logging.log(FINE, f"{step_limiter:.2g} -> !! good !! (stepd {np.linalg.norm(actual_step):.3g})")
					break
				elif new_value < value:
					logging.log(FINE, f"{step_limiter:7.2g} -> .. not better enuff")
				else:
					logging.log(FINE, f"{step_limiter:7.2g} -> .. not better")

			# if the condition is not met, decrement the step size and try agen
			step_limiter *= STEP_REDUCTION
			num_step_sizes += 1
			# keep track of the number of step sizes we've tried
			if num_step_sizes > 100:
				raise RuntimeError("line search did not converge")

		# do a few final calculations
		gradient_magnitude = np.linalg.norm(gradient)
		gradient_angle = np.linalg.norm(actual_step)/np.linalg.norm(ideal_step)
		report(state, value, gradient, actual_step, gradient_angle)

		# if the termination condition is met, finish
		if gradient_magnitude*gradient_angle < gradient_tolerance:
			logging.log(INFO, f"Completed in {num_line_searches} iterations.")
			return state

		# take the new state and error value
		state = new_state
		value = new_value
		# recompute the gradient once per outer loop
		gradient, hessian = get_gradient(state)
		# set the step size back a bit
		step_limiter /= STEP_RELAXATION
		# keep track of the number of iterations
		num_line_searches += 1
		if num_line_searches >= 1e5:
			raise RuntimeError(f"algorithm did not converge in {num_step_sizes} iterations")

def polytope_project(point: NDArray[float],
                     polytope_mat: DenseSparseArray, polytope_lim: NDArray[float]
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
	return minimize_quadratic_in_polytope(
		point,
		DenseSparseArray.identity(point.shape),
		np.zeros(point.shape),
		polytope_mat, polytope_lim)


def minimize_quadratic_in_polytope(fixed_point: NDArray[float],
                                   hessian: DenseSparseArray,
                                   gradient: NDArray[float],
                                   polytope_mat: DenseSparseArray, polytope_lim: NDArray[float],
                                   return_unbounded_solution: bool = False,
                                   tolerance: float = 1e-8,
                                   iterations_per_iteration: int = 1,
                                   ) -> NDArray[float] | tuple[NDArray[float], NDArray[float]]:
	""" find the global extremum of the concave-up multivariate quadratic function:
	        f(x) = (x - x0)⋅(hessian + additional_convexity*I)@(x - x0) + gradient⋅(x - x0)
	    subject to the inequality constraint
	        all(polytope_mat@point <= polytope_lim + tolerance)
	    using the alternating-direction method of multipliers, as is thoroly described in
	        S. Boyd, N. Parikh, E. Chu, B. Peleato, and J. Eckstein. "Distributed Optimization and
	        Statistical Learning via the Alternating Direction Method of Multipliers".
	        *Foundations and Trends in Machine Learning* Vol. 3 No. 1 (2010) 1–122.
	        DOI: 10.1561/2200000016
	    :param fixed_point: the point at which the quadratic function is defined; generally the
	                        quadratic function is a Taylor expansion, and this will be the point
	                        about which Taylor is expanding
	    :param hessian: the primary twoth-derivative matrix of the quadratic function at the fixed
	                    point. it must be symmetric and should be positive definite.
	    :param gradient: the gradient of the quadratic function at the fixed point.
	    :param polytope_mat: the normal vectors of the polytope faces, by which the polytope is defined
	    :param polytope_lim: the quantity that defines the size of the polytope. a point is in the polytope iff
	                         polytope_mat @ point[:, k] <= polytope_lim[k] for all k
	    :param return_unbounded_solution: whether to also return the solution if there were no bounds
	    :param tolerance: the relative tolerance at which to terminate iterations
	    :param iterations_per_iteration: the number of times x and z are updated for each time y is
	                                      updated.  I don't know what this is supposed to be.
	    :return: the bounded solution, and also -- if return_unbounded_solution is true -- the unbounded solution
	"""
	if polytope_mat.ndim != 2:
		raise ValueError(f"the matrix should be a (2d) matrix, not a {polytope_mat.ndim}d-array")

	if iterations_per_iteration < 1:
		raise ValueError("you must have at least one iteration per iteration")

	if not hessian.is_positive_definite():
		raise ConcaveObjectiveFunctionException("the given matrix is not positive definite")

	# check to see if we're already done
	x_unbounded = fixed_point - hessian.inverse_matmul(gradient)
	if hessian.is_positive_definite() and np.all(polytope_mat@x_unbounded <= polytope_lim):
		if return_unbounded_solution:
			return x_unbounded, x_unbounded
		else:
			return x_unbounded

	# choose some initial gesses
	x_old = crudely_polytope_project(x_unbounded, polytope_mat, polytope_lim)
	z_old = polytope_mat@x_old
	y_old = np.zeros(z_old.shape)

	# redefine the gradient at the origin (so that we may discard fixed_point)
	gradient = gradient - hessian@fixed_point

	# compute this matrix (it's simple in principle but kind of annoying in practice)
	polytope_mat_square = polytope_mat.T@polytope_mat
	if z_old.ndim > 1:
		polytope_mat_square = polytope_mat_square.repeat_diagonally(z_old.shape[1:])

	# establish the parameters and persisting variables
	primal_tolerance = tolerance*np.linalg.norm(polytope_lim)
	dual_tolerance = primal_tolerance*sqrt(x_old.size/z_old.size)
	step_size = 1#np.linalg.norm(hessian.diagonal())/np.linalg.norm(polytope_mat.transpose_matmul_self().diagonal())
	num_iterations = 0

	# loop thru the alternating direction multiplier method
	history = []
	while True:
		# update x, z, and then y
		for i in range(iterations_per_iteration):
			total_hessian = hessian + polytope_mat_square*step_size  # TODO: only recompute this when step_size changes
			total_gradient = gradient - polytope_mat.T@(step_size*z_old - y_old)
			x_new = total_hessian.inverse_matmul(-total_gradient)#, guess=x_old)
			polytope_mat_x_new = polytope_mat@x_new
			z_new = np.minimum(polytope_lim, polytope_mat_x_new + y_old/step_size)
		y_new = y_old + step_size*(polytope_mat_x_new - z_new)

		# check the stopping criteria
		primal_residual = np.linalg.norm(polytope_mat_x_new - z_new)
		dual_residual = step_size*np.linalg.norm(polytope_mat.T@(z_new - z_old))
		if primal_residual <= primal_tolerance and dual_residual <= dual_tolerance:
			print(primal_residual, primal_tolerance, dual_residual, dual_tolerance)
			x_final = crudely_polytope_project(x_new, polytope_mat, polytope_lim)
			print(num_iterations, end=": ")
			if return_unbounded_solution:
				return x_final, x_unbounded
			else:
				return x_final

		history.append([primal_residual, dual_residual, 1/2*np.sum(x_new*(hessian@x_new)) + np.sum(x_new*gradient), step_size])

		# adjust the step size for the next iteration
		# if primal_residual/primal_tolerance > 8*dual_residual/dual_tolerance:
		# 	step_size *= 2
		# elif dual_residual/dual_tolerance > 8*primal_residual/primal_tolerance:
		# 	step_size /= 2

		# carry over the loop variables
		x_old, z_old, y_old = x_new, z_new, y_new
		# make sure it doesn't run for too long
		num_iterations += 1
		if num_iterations >= 10_000/iterations_per_iteration:
			print(num_iterations, end=": ")
			import matplotlib.pyplot as plt
			plt.figure()
			plt.plot(history)
			plt.axhline(primal_tolerance)
			plt.axhline(dual_tolerance)
			plt.yscale('log')
			print(history)
			plt.show()
			raise MaxIterationsException("The maximum number of iterations was reached in the alternating directions multiplier method")


def crudely_polytope_project(point, polytope_mat, polytope_lim):
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
	polytope_magnitudes = (polytope_mat**2).sum(axis=1)
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


if __name__ == "__main__":
	import matplotlib.pyplot as plt
	import numpy as np

	polytope_matrix = DenseSparseArray.from_coordinates(
		[2],
		np.array([[[0], [1]], [[0], [1]], [[0], [1]], [[0], [1]], [[0], [1]]]),
		np.array([[.7, .3], [0., 1.1], [0., -.8], [-.6, 0.], [-.7, -.7]]))
	polytope_limits = np.array(1.)

	X, Y = np.meshgrid(np.linspace(-2, 2, 101), np.linspace(-2, 2, 101), indexing="ij")

	point = np.array([2., 1.6])
	projection = polytope_project(point, polytope_matrix, polytope_limits)

	plt.contour(X, Y, np.max(polytope_matrix@np.stack([X, Y]), axis=0), levels=[polytope_limits], colors="k")
	plt.plot(point[0], point[1], "x")
	plt.plot(projection[0], projection[1], "o")
	print(projection)
	plt.axis("equal")
	plt.show()

	x0 = np.array([1., -1.])
	gradient = np.array([2.5, -3.0])
	hessian = DenseSparseArray.from_coordinates([2],
	                                            np.array([[[0], [1]], [[0], [1]]]),
	                                            np.array([[1.0, -0.9], [-0.9, 1.0]]))
	I = DenseSparseArray.identity(hessian.dense_shape)

	solutions = []
	for caution in [0, .01, .1, 1, 10, 100, 10000]:
		print(f"using caution of {caution}")
		solution = minimize_quadratic_in_polytope(x0, hessian + I*caution, gradient,
		                                          polytope_matrix, polytope_limits,
		                                          return_unbounded_solution=False)
		print("done!\n\n")
		solutions.append(solution)

	plt.contour(X, Y, np.max(polytope_matrix@np.stack([X, Y]), axis=0), levels=[polytope_limits], colors="k")
	dX, dY = X - x0[0], Y - x0[1]
	plt.contourf(X, Y, dX*gradient[0] + dY*gradient[1] +
	             1/2*(dX**2*np.array(hessian)[0, 0] + 2*dX*dY*np.array(hessian)[0, 1] + dY**2*np.array(hessian)[1, 1]))
	plt.axis("equal")

	plt.plot(x0[0], x0[1], "wo")
	plt.plot([p[0] for p in solutions], [p[1] for p in solutions], "w-x")
	plt.show()
