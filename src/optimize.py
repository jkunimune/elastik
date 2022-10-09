#!/usr/bin/env python
"""
optimize.py

minimize an objective function using gradient descent with a simple line-search to ensure
it doesn't overshoot. there are probably scipy functions that do this, but I don't know
what the name for this algorithm would be, and anyway, I want to be able to see its
progress as it goes.
"""
from __future__ import annotations

from math import sqrt
from typing import Callable

import numpy as np
from numpy._typing import NDArray

from sparse import DenseSparseArray

STEP_REDUCTION = 5.
STEP_AUGMENTATION = STEP_REDUCTION**2.6
STEP_MAXIMUM = 1e3
LINE_SEARCH_STRICTNESS = (STEP_REDUCTION - 1)/(STEP_REDUCTION**2 - 1)


class Variable:
	def __init__(self, values: NDArray[float] | Variable,
	             gradients: NDArray[float] = None,
	             curvatures: NDArray[float] = None,
	             independent: bool = False, ndim: int = 0):
		""" an array of values with gradient information attached, for computing gradients
		    of vectorized functions
			:param values: the local value of the quantity
			:param gradients: the gradients of the values with respect to some basis. if
			                  none are specified, the values are assumed to be the
			                  independent basis variables, and the gradients are set to
			                  orthogonal unit vectors. unless independent is False; then
			                  they're just zero.
			:param curvatures: the diagonals of the hessian with respect to some basis.
			                   if none are specified, it is assumed to be all zero.
		    :param independent: whether the gradient should be set to an identity matrix
		                        (otherwise it's zero)
		    :param ndim: the minimum number of dimensions for the values. if
		                           the provided values have fewer dimensions than this,
		                           then 1s will be added to the end of the shape. it's
		                           mostly useful for converting scalar constants.
		"""
		if type(values) == Variable:
			if gradients is not None:
				raise ValueError("You must not supply gradients when the first argument is already a Variable")
			self.values = values.values
			self.gradients = values.gradients
			self.curvatures = values.curvatures

		else:
			# ensure the values have at least num_dimensions dimensions
			self.values = np.reshape(values,
			                         np.shape(values) + (1,)*(ndim - np.ndim(values)))
			# if gradients are specified
			if gradients is not None:
				# make sure the shapes match
				if gradients.shape[:self.values.ndim] != self.values.shape:
					raise IndexError(f"the given array dimensions do not match (you passd {self.values.shape} values and {gradients.shape} gradients).")
				self.gradients = gradients
			# if no gradients are given and these are independent variables
			elif independent:
				# make an identity matrix of sorts
				self.gradients = DenseSparseArray.identity(self.values.shape) # TODO: if the gradient calculation step becomes very slow, I should try using sparse matrices for this part
			# if no gradients are given and these are not independent
			else:
				# take the values to be constant
				self.gradients = np.array(0)
			# if curvatures are given
			if curvatures is not None:
				if curvatures.shape != self.gradients.shape:
					raise IndexError("the given array dimensions do not match")
				self.curvatures = curvatures
			elif self.gradients.ndim > 0:
				self.curvatures = DenseSparseArray.zeros(
					self.values.shape, self.gradients.shape[self.values.ndim:])
			else:
				self.curvatures = np.array(0)

		self.shape = self.values.shape
		""" the shape of self.values """
		self.space = self.gradients.shape[self.values.ndim:]
		""" the shape of each gradient """
		self.bc = (slice(None),)*len(self.shape) + (np.newaxis,)*len(self.space)
		""" this tuple should be used to index self.values when they need to broadcast
		    to the shape of self.gradients
		"""

		self.ndim = len(self.shape)
		self.size = np.product(self.shape)

	def __str__(self):
		return f"{'x'.join(str(i) for i in self.shape)}({'x'.join(str(i) for i in self.space)})"

	def __getitem__(self, item):
		if type(item) is not tuple:
			item = (item,)
		value_index = item
		gradient_index = (slice(None),)*len(self.space)
		return Variable(self.values[value_index],
		                self.gradients[(*value_index, *gradient_index)],
		                self.curvatures[(*value_index, *gradient_index)])

	def __add__(self, other):
		other = Variable(other)
		return Variable(self.values + other.values,
		                self.gradients + other.gradients,
		                self.curvatures + other.curvatures)

	def __le__(self, other):
		other = Variable(other, ndim=self.ndim)
		return self.values <= other.values

	def __lt__(self, other):
		other = Variable(other, ndim=self.ndim)
		return self.values < other.values

	def __ge__(self, other):
		other = Variable(other, ndim=self.ndim)
		return self.values >= other.values

	def __gt__(self, other):
		other = Variable(other, ndim=self.ndim)
		return self.values > other.values

	def __mul__(self, other):
		other = Variable(other, ndim=self.ndim)
		return Variable(values=self.values * other.values,
		                gradients=self.gradients * other.values[self.bc] +
		                          other.gradients * self.values[self.bc],
		                curvatures=self.curvatures * other.values[self.bc] +
		                           self.gradients * other.gradients * 2 +
		                           other.curvatures * self.values[self.bc])

	def __neg__(self):
		return self * (-1)

	def __pow__(self, power):
		return Variable(self.values ** power,
		                self.gradients * self.values[self.bc]**(power - 1) * power,
		                (self.gradients**2 * (power - 1) +
		                 self.curvatures * self.values[self.bc]) *
		                self.values[self.bc]**(power - 2) * power)

	def __sub__(self, other):
		other = Variable(other)
		return Variable(self.values - other.values,
		                self.gradients - other.gradients,
		                self.curvatures - other.curvatures)

	def __truediv__(self, other):
		return self * other**(-1)

	def __rtruediv__(self, other):
		return other * self**(-1)

	def __radd__(self, other):
		return self + other

	def __rsub__(self, other):
		return -self + other

	def __rmatmul__(self, other):
		return Variable(other@self.values, other@self.gradients, other@self.curvatures)

	def __rmul__(self, other): # watch out! never multiply ndarray*Variable, as I can't figure out how to override Numpy's bad behavior there
		return self * other

	def sqrt(self):
		return self ** 0.5

	def log(self):
		return Variable(np.log(self.values),
		                self.gradients / self.values[self.bc],
		                self.curvatures / self.values[self.bc] - self.gradients**2 / self.values[self.bc]**2)

	def exp(self):
		return Variable(np.exp(self.values),
		                self.gradients * np.exp(self.values[self.bc]),
		                (self.curvatures + self.gradients**2) * np.exp(self.values[self.bc]))

	def sum(self, axis=None):
		if axis is None:
			axis = tuple(np.arange(self.ndim))
		else:
			axis = np.atleast_1d(axis)
			axis = (axis + self.ndim)%self.ndim
			axis = tuple(axis)
		return Variable(self.values.sum(axis=axis),
		                self.gradients.sum(axis=axis),
		                self.curvatures.sum(axis=axis))


def minimize(func: Callable[[NDArray[float] | Variable], float | Variable],
             guess: NDArray[float],
             tolerance: float,
             bounds_matrix: DenseSparseArray = None,
             bounds_limits: NDArray[float] | list[float] = None,
             report: Callable[[NDArray[float], float, NDArray[float], NDArray[float], bool], None] = None,
             backup_func: Callable[[NDArray[float] | Variable], float | Variable] = None,
             ) -> np.ndarray:
	""" find the vector that minimizes a function of a list of points using gradient
	    descent with a dynamically chosen step size. unlike a more generic minimization
	    function, this one assumes that each datum is a vector, not a scalar, so many
	    things have one more dimension that you might otherwise expect.
	    :param func: the objective function to minimize. it takes an array of size n×2 as
	                 argument and returns a single scalar value
	    :param guess: the initial input to the function, from which the gradients will descend.
	    :param tolerance: the absolute tolerance. when the magnitude of the gradient at
	                      any given point dips below this, we are done.
	    :param bounds_matrix: a list of inequality constraints on various linear combinations.
	                          it should be some object that matrix-multiplies by the state array to
	                          produce a m×2 vector of tracer particle positions
	    :param bounds_limits: the values of the inequality constraints. should be a 2-vector
	                          representing the maximum allowable x and y coordinates of those tracer
	                          particles.
	    :param report: an optional function that will be called each time a line search
	                   is completed, to provide real-time information on how the fitting
	                   routine is going. it takes as arguments the current state, the
	                   current value of the function, the current gradient, the previous
	                   step if any, and whether this is the final value
	    :param backup_func: an optional additional objective function to use when func is
	                        nonapplicable. specificly, when the primary objective function
	                        is only defined in a certain domain but the initial guess may
	                        be outside of it, the backup can be used to push the state
	                        vector into that domain. it should return smaller and smaller
	                        values as the state approaches the valid domain and -inf for
	                        states inside it. if a -inf in achieved with the backup
	                        function, it will immediately switch to the primary function.
	                        if -inf is never returned and the backup function converges,
	                        that minimum will be returnd.
	    :return: the optimal n×2 array of points
	"""
	# start by checking the guess agenst the bounds
	if bounds_matrix is None:
		if bounds_limits is not None:
			raise ValueError("you mustn't pass bounds_limits without bounds_matrix")
		bounds_tolerance = None
	else:
		if bounds_limits is None:
			raise ValueError("you mustn't pass bounds_matrix without bounds_limits")
		guess = polytope_project(guess, bounds_matrix, bounds_limits, tolerance=0, certainty=60)
		bounds_tolerance = np.min(bounds_limits)*1e-3
		if np.all(np.isinf(bounds_limits)): # go ahead and remove any pointless bounds
			bounds_matrix = None
			bounds_limits = None

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
			raise RuntimeError("there are nan values")
		return value
	# define a utility function to use Variable to get the gradient of the value
	def get_gradient(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
		variable = func(Variable(x, independent=True))
		if np.any(np.isnan(variable.gradients)):
			raise RuntimeError("there are nan gradients")
		return variable.gradients, np.sum(variable.curvatures, axis=-1)

	initial_value = get_value(guess)

	# check just in case we instantly fall thru to the followup function
	if initial_value == -np.inf and followup_func is not None:
		func = followup_func
		followup_func = None
		initial_value = get_value(guess)
	elif not np.isfinite(initial_value):
		raise RuntimeError(f"the objective function returned an invalid initial value: {initial_value}")

	# instantiate the loop state variables
	fast_mode = True
	value = initial_value
	state = guess
	# and with the step size parameter set
	step_size = 1e2#STEP_MAXIMUM
	# descend until we can't descend any further
	num_line_searches = 0
	while True:
		# compute the gradient once per outer loop
		gradient, curvature = get_gradient(state)
		assert gradient.shape == state.shape

		# descend the gradient
		if fast_mode: # well, gradient scaled with curvature if possible
			curvature_cutoff = np.quantile(abs(curvature), .01)
			direction = -gradient/np.maximum(curvature, curvature_cutoff)[:, np.newaxis]
		else:
			direction = -gradient

		# do a line search to choose a good step size
		num_step_sizes = 0
		while True:
			# step according to the gradient and step size for this inner loop
			new_state = state + step_size*direction
			# projecting onto the legal subspace if necessary
			if bounds_limits is not None:
				new_state = polytope_project(
					new_state, bounds_matrix, bounds_limits, bounds_tolerance)
			step = new_state - state
			new_value = get_value(new_state)
			# if we're going in the wrong direction, disable fast mode and try again
			if np.sum(step*gradient) > 0:
				assert fast_mode, "projection caused it to step uphill"
				fast_mode = False
				new_state, new_value = state, value
				break
			# if this is infinitely good, jump to the followup function now
			if new_value == -np.inf and followup_func is not None:
				print(f"Reached the valid domain in {num_line_searches} iterations.")
				return minimize(followup_func, new_state, tolerance, bounds_matrix, bounds_limits, report, None)
			# if the line search condition is met, take it
			if new_value < value + LINE_SEARCH_STRICTNESS*np.sum(step*gradient):
				break
			# if the condition is not met, decrement the step size and try agen
			step_size /= STEP_REDUCTION
			num_step_sizes += 1
			# keep track of the number of step sizes we've tried
			if num_step_sizes > 100:
				raise RuntimeError("line search did not converge")

		# if the termination condition is met, finish
		if -np.sum(gradient*step)/np.linalg.norm(step) > tolerance:
			report(state, value, gradient, step, False)
		else:
			report(state, value, gradient, step, True)
			print(f"Completed in {num_line_searches} iterations.")
			if bounds_limits is not None:
				state = polytope_project(state, bounds_matrix, bounds_limits, 0, 100)
			return state

		# take the new state and error value
		state = new_state
		value = new_value
		# set the step size back a bit
		step_size *= STEP_AUGMENTATION
		# keep track of the number of iterations
		num_line_searches += 1
		if num_line_searches > 1e5:
			raise RuntimeError(f"algorithm did not converge in {num_step_sizes} iterations")


def polytope_project(point: NDArray[float], polytope_mat: DenseSparseArray, polytope_lim: float | NDArray[float],
                     tolerance: float, certainty: float = 20) -> NDArray[float]:
	""" project a given point onto a polytope defined by the inequality
	        all(ploytope_mat@point <= polytope_lim + tolerance)
	    I learned this fast dual-based proximal gradient strategy from
	        Beck, A. & Teboulle, M. "A fast dual proximal gradient algorithm for
	        convex minimization and applications", <i>Operations Research Letters</i> <b>42</b> 1
	        (2014), p. 1–6. doi:10.1016/j.orl.2013.10.007,
	    :param point: the point to project
	    :param polytope_mat: the matrix that defines the normals of the polytope faces
	    :param polytope_lim: the quantity that defines the size of the polytope.  it may be an array
	                         if point is 2d.  a point is in the polytope iff
	                         polytope_mat @ point[:, k] <= polytope_lim[k] for all k
	    :param tolerance: how far outside of the polytope a returned point may be
	    :param certainty: how many iterations it should try once it's within the tolerance to ensure
	                      it's finding the best point
	"""
	if point.ndim == 2:
		if point.shape[1] != polytope_lim.size:
			raise ValueError(f"if you want this fancy functionality, the shapes must match")
		# if there are multiple dimensions, do each dimension one at a time
		return np.stack([polytope_project(point[:, k], polytope_mat, polytope_lim[k], tolerance, certainty)
		                 for k in range(point.shape[1])]).T
	elif point.ndim != 1 or np.ndim(polytope_lim) != 0:
		raise ValueError(f"I don't think this works with {point.ndim}d-arrays instead of (1d) vectors")
	if polytope_mat.ndim != 2:
		raise ValueError(f"the matrix should be a (2d) matrix, not a {polytope_mat.ndim}d-array")
	if point.shape[0] != polytope_mat.shape[1]:
		raise ValueError("these polytope definitions don't jive")
	# check to see if we're already done
	if np.all(polytope_mat@point <= polytope_lim):
		return point

	# establish the parameters and persisting variables
	L = np.linalg.norm(polytope_mat, ord=2)**2
	x_new = None
	w_old = y_old = np.zeros(polytope_mat.shape[0])
	t_old = 1
	candidates = []
	# loop thru the proximal gradient descent of the dual problem
	for i in range(10_000):
		grad_F = polytope_mat@(point + polytope_mat.T@w_old)
		prox_G = np.minimum(polytope_lim, grad_F - L*w_old)
		y_new = w_old - (grad_F - prox_G)/L
		t_new = (1 + sqrt(1 + 4*t_old**2))/2  # this inertia term is what makes it fast
		w_new = y_new + (t_old - 1)/t_new*(y_new - y_old)
		x_new = point + polytope_mat.T@y_new
		# save any points that are close enuff to being in
		if np.all(polytope_mat@x_new <= polytope_lim + tolerance):
			candidates.append(x_new)
			# and terminate once we get enuff of them
			if len(candidates) >= certainty:
				best = np.argmin(np.linalg.norm(np.array(candidates) - point, axis=1))
				return candidates[best]
		t_old, w_old, y_old = t_new, w_new, y_new
	return x_new

if __name__ == "__main__":
	import matplotlib.pyplot as plt
	x = Variable(np.linspace(0, 1, 26), np.ones((26, 1)), np.zeros((26, 1)))
	y = (x - 3)*x
	plt.plot(x.values, y.values, label="value")
	plt.plot(x.values, y.gradients, label="dy/dx")
	plt.plot(x.values, y.curvatures, label="d2y/dx2")
	plt.legend()
	plt.show()
