#!/usr/bin/env python
"""
optimize.py

minimize an objective function using gradient descent with a simple line-search to ensure
it doesn't overshoot. there are probably scipy functions that do this, but I don't know
what the name for this algorithm would be, and anyway, I want to be able to see its
progress as it goes.
"""
from __future__ import annotations

from math import inf
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from sparse import DenseSparseArray
from util import polytope_project

STEP_REDUCTION = 5.
STEP_AUGMENTATION = STEP_REDUCTION**2
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
				self.gradients = DenseSparseArray.identity(self.values.shape)
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
             gradient_tolerance: float,
             cosine_tolerance: float,
             bounds_matrix: DenseSparseArray = None,
             bounds_limits: NDArray[float] | list[float] = None,
             report: Callable[[NDArray[float], float, NDArray[float], NDArray[float], float], None] = None,
             backup_func: Callable[[NDArray[float] | Variable], float | Variable] = None,
             ) -> np.ndarray:
	""" find the vector that minimizes a function of a list of points using gradient
	    descent with a dynamically chosen step size. unlike a more generic minimization
	    function, this one assumes that each datum is a vector, not a scalar, so many
	    things have one more dimension that you might otherwise expect.
	    :param func: the objective function to minimize. it takes an array of size n×2 as
	                 argument and returns a single scalar value
	    :param guess: the initial input to the function, from which the gradients will descend.
	    :param gradient_tolerance: the absolute tolerance. if the magnitude of the gradient at any
	                               given point dips below this, we are done.
	    :param cosine_tolerance: the relative tolerance. if the normalized dot-product of the
	                             gradient and the projected step direction dips below this, we done.
	    :param bounds_matrix: a list of inequality constraints on various linear combinations.
	                          it should be some object that matrix-multiplies by the state array to
	                          produce a m×2 vector of tracer particle positions
	    :param bounds_limits: the values of the inequality constraints. should be a 2-vector
	                          representing the maximum allowable x and y coordinates of those tracer
	                          particles.
	    :param report: an optional function that will be called each time a line search is
	                   completed, to provide real-time information on how the fitting routine is
	                   going. it takes as arguments the current state, the current value of the
	                   function, the current gradient magnitude, the previous step, and the ratio
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
		bounds_tolerance = np.min(bounds_limits)*1e-4
		if np.all(np.isinf(bounds_limits)): # go ahead and remove any pointless bounds
			bounds_matrix = None
			bounds_limits = None
	if bounds_limits is None:
		bounds_mode = False
		cosine_tolerance = -inf
	else:
		bounds_mode = True

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
	def get_gradient(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
		variable = func(Variable(x, independent=True))
		if np.any(np.isnan(variable.gradients)):
			raise RuntimeError(f"there are nan gradients at x = {x}")
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
	value = initial_value
	state = guess
	# and with the step size parameter set
	step_size = STEP_MAXIMUM
	# descend until we can't descend any further
	num_line_searches = 0
	while True:
		# compute the gradient once per outer loop
		gradient, curvature = get_gradient(state)
		assert gradient.shape == state.shape

		# descend the gradient
		if bounds_mode:
			direction = -gradient
		else:  # well, gradient scaled with curvature if possible
			curvature_cutoff = np.quantile(abs(curvature), .01)
			direction = -gradient/np.maximum(curvature, curvature_cutoff)[:, np.newaxis]

		# do a line search to choose a good step size
		num_step_sizes = 0
		while True:
			# step according to the gradient and step size for this inner loop
			new_state = state + step_size*direction
			# projecting onto the legal subspace if necessary
			if bounds_mode:
				new_state = polytope_project(
					new_state, bounds_matrix, bounds_limits, bounds_tolerance)
			step = new_state - state
			new_value = get_value(new_state)
			if np.sum(step*gradient) > 0:
				if bounds_mode:
					print("Warning: more precise polytope projection mite be required")
					break
				else:
					raise RuntimeError("this problem needed to be run in bounds mode, I gess...")
			# if this is infinitely good, jump to the followup function now
			if new_value == -np.inf and followup_func is not None:
				print(f"Reached the valid domain in {num_line_searches} iterations.")
				return minimize(followup_func, new_state, gradient_tolerance, cosine_tolerance,
				                bounds_matrix, bounds_limits, report, None)
			# if the line search condition is met, take it
			if new_value < value + LINE_SEARCH_STRICTNESS*np.sum(step*gradient):
				break
			# if the condition is not met, decrement the step size and try agen
			step_size /= STEP_REDUCTION
			num_step_sizes += 1
			# keep track of the number of step sizes we've tried
			if num_step_sizes > 100:
				raise RuntimeError("line search did not converge")

		# do a few final calculations
		gradient_magnitude = np.linalg.norm(gradient)
		if bounds_mode:
			gradient_angle = np.sum(direction*step/step_size)/np.sum(direction**2)
		else:
			gradient_angle = 1
		report(state, value, gradient, step, gradient_angle)

		# if the termination condition is met, finish
		if gradient_magnitude < gradient_tolerance or gradient_angle < cosine_tolerance:
			print(f"Completed in {num_line_searches} iterations.")
			if bounds_mode:
				state = polytope_project(state, bounds_matrix, bounds_limits, 0, 100)
			return state

		# take the new state and error value
		state = new_state
		value = new_value
		# set the step size back a bit
		step_size *= STEP_AUGMENTATION
		# keep track of the number of iterations
		num_line_searches += 1
		if num_line_searches >= 1e5:
			raise RuntimeError(f"algorithm did not converge in {num_step_sizes} iterations")


if __name__ == "__main__":
	import matplotlib.pyplot as plt
	x = Variable(np.linspace(0, 1, 26), np.ones((26, 1)), np.zeros((26, 1)))
	y = (x - 3)*x
	plt.plot(x.values, y.values, label="value")
	plt.plot(x.values, y.gradients, label="dy/dx")
	plt.plot(x.values, y.curvatures, label="d2y/dx2")
	plt.legend()
	plt.show()
