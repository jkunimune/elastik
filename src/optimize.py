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


class Variable:
	def __init__(self, values: np.ndarray or "Variable", gradients: np.ndarray = None,
	             independent: bool = False, num_dimensions: int = 0):
		""" an array of values with gradient information attached, for computing gradients
		    of vectorized functions
			:param values: the local value of the quantity
			:param gradients: the gradients of the values with respect to some basis. if
			                  none are specified, the values are assumed to be the
			                  independent basis variables, and the gradients are set to
			                  orthogonal unit vectors.
		    :param independent: whether the gradient should be set to an identity matrix
		                        (otherwise it's zero)
		    :param num_dimensions: the minimum number of dimensions for the values. if
		                           the provided values have fewer dimensions than this,
		                           then 1s will be added to the end of the shape. it's
		                           mostly useful for converting scalar constants.
		"""
		if type(values) == Variable:
			if gradients is not None:
				raise ValueError("You must not supply gradients when the first argument is already a Variable")
			self.values = values.values
			self.gradients = values.gradients

		else:
			# ensure the values have at least num_dimensions dimensions
			self.values = np.reshape(values,
				np.shape(values) + (1,)*(num_dimensions - len(np.shape(values))))
			# if gradients are specified
			if gradients is not None:
				# make sure the shapes match
				if gradients.shape[:len(self.values.shape)] != self.values.shape:
					raise IndexError("the given array dimensions do not match.")
				self.gradients = gradients
			# if no gradients are given and these are independent variables
			elif independent:
				# make an identity matrix of sorts
				self.gradients = np.identity(self.values.size).reshape(self.values.shape*2)
			# if no gradients are given and these are not independent
			else:
				# take the values to be constant
				self.gradients = np.array(0)

		self.shape = self.values.shape
		""" the shape of self.values """
		self.space = self.gradients.shape[len(self.values.shape):]
		""" the shape of each gradient """
		self.bc = (slice(None),)*len(self.shape) + (None,)*len(self.space)
		""" this tuple should be used to index self.values when they need to broadcast
		    to the shape of self.gradients
		"""

	def __getitem__(self, item):
		return Variable(self.values[item], self.gradients[(*item, ...)])

	def __add__(self, other):
		other = Variable(other)
		return Variable(self.values + other.values,
		                self.gradients + other.gradients)

	def __mul__(self, other):
		other = Variable(other, num_dimensions=len(self.shape))
		return Variable(self.values * other.values,
		                self.gradients * other.values[self.bc] +
		                self.values[self.bc] * other.gradients)

	def __neg__(self):
		return self * (-1)

	def __pow__(self, power):
		return Variable(self.values ** power,
		                self.gradients * self.values[self.bc]**(power - 1) * power)

	def __sub__(self, other):
		other = Variable(other)
		return Variable(self.values - other.values,
		                self.gradients - other.gradients)

	def __truediv__(self, other):
		return self * other**(-1)

	def __radd__(self, other):
		return self + other

	def __rsub__(self, other):
		return -self + other

	def __rmul__(self, other):
		return self * other

	def sqrt(self):
		return self ** 0.5

	def sum(self, axis=None):
		if axis is None:
			axis = tuple(np.arange(len(self.shape)))
		else:
			axis = tuple(np.array(axis))
			axis = (axis + len(self.shape))%len(self.shape)
		return Variable(self.values.sum(axis=axis),
		                self.gradients.sum(axis=axis))


def minimize(func: Callable[[np.ndarray or Variable], float or Variable],
             guess: np.ndarray,
             scale: np.ndarray = None,
             tolerance: float = 1e-8,
             bounds: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
             report: Callable[[np.ndarray, float], None] = None,
             ) -> np.ndarray:
	""" find the vector that minimizes a function of a list of points using gradient
	    descent with a dynamically chosen step size. unlike a more generic minimization
	    function, this one assumes that each datum is a vector, not a scalar, so many
	    things have one more dimension that you might otherwise expect.
	    :param func: the objective function to minimize. it takes an array of size n×2 as
	                 argument and returns a single scalar value
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
	    :param report: an optional function that will be called each time a line search
	                   is completed, to provide real-time information on how the fitting
	                   routine is going. it takes as arguments the current state and
	                   current value of the function
	    :return: the optimal n×2 array of points
	"""
	n, d = guess.shape
	if scale is None:
		scale = np.ones(n)
	if bounds is None:
		bounds = []

	# redefine the objective function to have some checks bilt in
	def get_value(x: np.ndarray) -> float:
		value = func(x)
		if value <= 0:
			raise ValueError("I'm not set up to have objective functions that can go nonpositive")
		return value
	# define a utility function to use Variable to get the gradient of the value
	def get_gradient(x: np.ndarray) -> np.ndarray:
		return func(Variable(x, independent=True)).gradients

	# start at the inicial gess
	state = guess
	value = get_value(state)
	# and with the step size parameter set to unity
	step_size = 1.
	# descend until we can't descend any further
	num_line_searches = 0
	while True:
		report(state, value)
		# compute the gradient once per outer loop
		direction = -get_gradient(state)
		assert direction.shape == state.shape
		# do a line search to choose a good step size
		num_step_sizes = 0
		while True:
			# step according to the gradient and step size for this inner loop
			new_state = state + direction*step_size
			new_value = get_value(new_state)
			# if the line search condition is met, take it
			if new_value < value:
				break
			# if the condition is not met, decrement the step size and try agen
			step_size /= 5.
			# keep track of the number of step sizes we've tried
			if num_step_sizes > 100:
				raise RuntimeError("line search did not converge")
		# if the termination condition is met, finish
		if (value - new_value)/value < tolerance:
			return new_state
		# take the new state and error value
		state = new_state
		value = new_value
		# and set the step size back a bit
		step_size *= 50.
		# keep track of the number of iterations
		num_line_searches += 1
		if num_line_searches > 1e6:
			raise RuntimeError(f"algorithm did not converge in {num_step_sizes} iterations")
