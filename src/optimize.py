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
from math import inf, sqrt, isfinite
from typing import Callable, Optional, Sequence

import numpy as np
from numpy.typing import NDArray

from autodiff import Variable
from sparse import DenseSparseArray

STEP_REDUCTION = 5.
STEP_RELAXATION = STEP_REDUCTION**1.5
LINE_SEARCH_STRICTNESS = (STEP_REDUCTION - 1)/(STEP_REDUCTION**2 - 1)

np.seterr(under="ignore", over="raise", divide="raise", invalid="raise")


class MaxIterationsException(Exception):
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
		if np.isnan(value):
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
	if initial_value == -np.inf and followup_func is not None:
		func = followup_func
		followup_func = None
		initial_value = get_value(guess)
	elif not np.isfinite(initial_value):
		raise RuntimeError(f"the objective function returned an invalid initial value: {initial_value}")

	# calculate the initial gradient
	gradient, hessian = get_gradient(guess)
	if gradient.shape != guess.shape:
		raise ValueError(f"the gradient function returned the wrong shape ({gradient.shape}, should be {guess.shape})")

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
				new_state, ideal_new_state = minimize_quadratic_in_polytope(
					state, hessian, step_limiter, gradient,
					bounds_matrix, bounds_limits,
					return_unbounded_solution=True)
			except MaxIterationsException:
				pass
				logging.log(19, f"{step_limiter:7.2g} -> xx not valid")
			else:
				ideal_step = ideal_new_state - state
				actual_step = new_state - state
				new_value = get_value(new_state)
				# if this is infinitely good, jump to the followup function now
				if new_value == -np.inf and followup_func is not None:
					logging.log(19, f"Reached the valid domain in {num_line_searches} iterations.")
					return minimize(followup_func, new_state, gradient_tolerance,
					                bounds_matrix, bounds_limits, report, None)
				# if the line search condition is met, take it
				if new_value < value + LINE_SEARCH_STRICTNESS*np.sum(actual_step*gradient):
					logging.log(19, f"{step_limiter:.2g} -> !! good")
					break
				elif new_value < value:
					logging.log(19, f"{step_limiter:7.2g} -> .. not better enuff")
				else:
					logging.log(19, f"{step_limiter:7.2g} -> .. not better")

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
			logging.log(20, f"Completed in {num_line_searches} iterations.")
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
		if point.shape[1] != polytope_lim.size:
			raise ValueError(f"if you want this fancy functionality, the shapes must match")
		# if there are multiple dimensions, do each dimension one at a time; it's faster, I think
		return np.stack([polytope_project(point[:, k], polytope_mat, polytope_lim[:, k])
		                 for k in range(point.shape[1])]).T
	elif point.ndim != 1:
		raise ValueError(f"I don't think this works with {point.ndim}d-arrays instead of (1d) vectors")
	return minimize_with_constraints(
		lambda x: 1/2*np.linalg.norm(point - x),
		lambda x: point - x,
		lambda z: np.all(z <= polytope_lim),
		lambda z: np.minimum(polytope_lim, z),
		polytope_mat, 1, point.shape)

def minimize_quadratic_in_polytope(fixed_point: NDArray[float],
                                   hessian: DenseSparseArray, damping: float,
                                   gradient: NDArray[float],
                                   polytope_mat: DenseSparseArray, polytope_lim: NDArray[float],
                                   return_unbounded_solution: bool,
                                   ) -> tuple[NDArray[float], NDArray[float]]:
	""" find the global extremum of the concave-up multivariate quadratic function:
	        f(x) = (x - x0)⋅(hessian + additional_convexity*I)@(x - x0) + gradient⋅(x - x0)
	    subject to the inequality constraint
	        all(polytope_mat@point <= polytope_lim + tolerance)
	    :param fixed_point: the point at which the quadratic function is defined; generally the
	                        quadratic function is a Taylor expansion, and this will be the point
	                        about which Taylor is expanding
	    :param hessian: the primary twoth-derivative matrix of the quadratic function at the fixed
	                    point. it must be symmetric and should be positive definite.
	    :param damping: a scalar term with which to augment the hessian matrix (hessian + additional_convexity*I)
	    :param gradient: the gradient of the quadratic function at the fixed point.
	    :param polytope_mat: the normal vectors of the polytope faces, by which the polytope is defined
	    :param polytope_lim: the quantity that defines the size of the polytope. a point is in the polytope iff
	                         polytope_mat @ point[:, k] <= polytope_lim[k] for all k
	    :param return_unbounded_solution: whether to also return the solution if there were no bounds
	    :return: the bounded solution, and also -- if return_unbounded_solution is true -- the unbounded solution
	"""
	if not return_unbounded_solution:
		raise NotImplementedError("no you can't do that")

	# if the damping is much larger than the hessian, save some time with this simpler calculation
	if damping > abs(hessian).max()*10:
		unbounded_solution = fixed_point - gradient/damping
		bounded_solution = polytope_project(unbounded_solution, polytope_mat, polytope_lim)
		return bounded_solution, unbounded_solution

	# calculate the eigen decomposition of the inverse hessian to save time later
	Λ, Q = hessian.symmetric_eigen_decomposition()
	Λ = np.maximum(0, Λ) + damping  # insert the damping here
	if np.any(Λ <= 0):
		raise ValueError("you need a nonzero step limiter if the hessian is not positive-definite")

	Λ = np.sqrt(Λ**2 + damping**2)**-1  # insert the damping here

	def func(x):
		dx = x - fixed_point
		quad_term = 1/2*np.sum(dx*(hessian@dx + damping*dx))
		lin_term = np.sum(gradient*dx)
		return quad_term + lin_term

	def unbounded_solution(x):
		argument = np.ravel(gradient + x)
		step = -Q@(Λ*(Q.T@argument))  # -hessian^-1 (gradient + x)
		step = np.reshape(step, fixed_point.shape)
		return fixed_point + step  # TODO I think conjugate gradients actually would be faster... but I would need another way to enforce convexity

	return (
		minimize_with_constraints(
			func, unbounded_solution,
			lambda z: np.all(z <= polytope_lim),
			lambda z: np.minimum(polytope_lim, z),
			polytope_mat,
			1/np.max(Λ),
			fixed_point.shape),
		unbounded_solution(0)
	)

def minimize_with_constraints(f: Callable[[NDArray[float]], float],
                              argmin_f: Callable[[NDArray[float]], NDArray[float]],
                              g: Callable[[NDArray[float]], bool],
                              prox_g: Callable[[NDArray[float]], NDArray[float]],
                              A: DenseSparseArray | NDArray[float],
                              σ: float, shape: Sequence[int], certainty: int = 30) -> NDArray[float]: # TODO: should it be 60?  would that work better?
	""" minimize a smooth multivariate function f(x) subject to a simple inequality constraint g(A@x)
	    I learned this fast dual-based proximal gradient strategy from
	        Beck, A. & Teboulle, M. "A fast dual proximal gradient algorithm for
	        convex minimization and applications", <i>Operations Research Letters</i> <b>42</b> 1
	        (2014), p. 1–6. doi:10.1016/j.orl.2013.10.007,
	    but I added a little twist.
	    that's not true.  I removed the twist because it wasn't very good, but then I couldn't bring
	    myself to remove that reference.
	    :param f: the smooth part of the function.  it must be continuusly differentiable or this won't work.
	    :param argmin_f: a function of d that returns the x that minimizes the expression f(x) + d⋅x
	    :param g: the constraint.  only values of x where g(A@x) is True are considered valid solutions
	    :param A: the matrix used to convert from real space to constraint space
	    :param prox_g: a function of z that returns the projection of z into the space where g(z)
	    :param σ: a lower bound on the convexity parameter of f
	    :param shape: the array shape that f and argmin_f expect
	    :param certainty: how many iterations it should try once it's within the tolerance to ensure
	                      it's finding the best point
	"""
	if A.ndim != 2:
		raise ValueError(f"the matrix should be a (2d) matrix, not a {A.ndim}d-array")
	if not isfinite(σ) or σ <= 0:
		raise ValueError(f"the strong convexity of f, by definition, must be positive.")

	def dual_to_primal(y):
		return argmin_f(-A.T@y)
	# check to see if we're already done
	y_old = np.zeros((A.shape[0],) + tuple(shape[1:]))
	x_guess = dual_to_primal(y_old)
	if g(A@x_guess):
		return x_guess

	# establish the parameters and persisting variables
	L = DenseSparseArray.linalg_norm(A, orde=2)**2/σ
	w_old = y_old
	t_old = 1
	candidates: list[tuple[float, NDArray[float]]] = []
	num_iterations = 0
	# loop thru the proximal gradient descent of the dual problem
	while True:
		# history.append(np.concatenate([y_old, dual_to_primal(y_old)]))
		u_new = argmin_f(-A.T@w_old)
		grad_F = A@u_new
		v_new = prox_g(grad_F - L*w_old)
		y_new = w_old - (grad_F - v_new)/L
		t_new = (1 + sqrt(1 + 4*t_old**2))/2  # this inertia term is what makes it fast
		w_new = y_new + (t_old - 1)/t_new*(y_new - y_old)
		x_new = dual_to_primal(y_new)
		# save the value of any point that meets the constraint
		if g(A@x_new):
			candidates.append((f(x_new), x_new))
			# and terminate once we get enough of them
			if len(candidates) >= certainty:
				_, best = min(candidates, key=lambda candidate: candidate[0])
				print(num_iterations, end=": ")
				return best
		t_old, w_old, y_old = t_new, w_new, y_new
		# make sure it doesn't run for too long
		num_iterations += 1
		if (num_iterations >= 50_000 and len(candidates) == 0) or num_iterations >= 80_000:
			print(num_iterations, end=": ")
			raise MaxIterationsException("The maximum number of iterations was reached in the fast dual-based proximal gradient routine")


if __name__ == "__main__":
	import matplotlib.pyplot as plt
	import numpy as np

	polytope_matrix = DenseSparseArray.from_coordinates(
		[2],
		np.array([[[0], [1]], [[0], [1]], [[0], [1]], [[0], [1]], [[0], [1]]]),
		np.array([[.7, .3], [0., 1.1], [0., -.8], [-.6, 0.], [-.7, -.7]]))
	polytope_limits = np.array(1.)

	X, Y = np.meshgrid(np.linspace(-2, 2, 101), np.linspace(-2, 2, 101), indexing="ij")
	plt.contour(X, Y, np.max(polytope_matrix@np.stack([X, Y]), axis=0), levels=[1.], colors="k")

	point = np.array([2., 1.6])
	projection = polytope_project(point, polytope_matrix, polytope_limits)
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

	plt.contour(X, Y, np.max(polytope_matrix@np.stack([X, Y]), axis=0), levels=[1.], colors="k")
	dX, dY = X - x0[0], Y - x0[1]
	plt.contourf(X, Y, dX*gradient[0] + dY*gradient[1] +
	             1/2*(dX**2*np.array(hessian)[0, 0] + 2*dX*dY*np.array(hessian)[0, 1] + dY**2*np.array(hessian)[1, 1]))
	plt.axis("equal")

	solutions = []
	for caution in [0, .01, .1, 1, 10, 100, 10000]:
		print(f"using caution of {caution}")
		solution, _ = minimize_quadratic_in_polytope(x0, hessian, caution, gradient,
		                                             polytope_matrix, polytope_limits,
		                                             return_unbounded_solution=True)
		print("done!\n\n")
		solutions.append(solution)
	plt.plot(x0[0], x0[1], "wo")
	plt.plot([p[0] for p in solutions], [p[1] for p in solutions], "w-x")
	plt.show()
