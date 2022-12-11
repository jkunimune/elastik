#!/usr/bin/env python
"""
autodiff.py

a file that defines a class that can be used in place of a numpy float array, but that will
automaticly calculate its own gradients and hessians when arithmetic operations are applied to it.
very useful for gradient- and hessian-based optimization rootenes.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from sparse import DenseSparseArray


class Variable:
	def __init__(self, values: NDArray[float] | Variable,
	             gradients: DenseSparseArray = None,
	             hessians: DenseSparseArray = None,
	             independent: bool = False, ndim: int = 0):
		""" an array of values with gradient information attached, for computing gradients and
		    hessians of vectorized functions
			:param values: the local value of the quantity
			:param gradients: the gradients of the values with respect to some basis. if
			                  none are specified, the values are assumed to be the
			                  independent basis variables, and the gradients are set to
			                  orthogonal unit vectors. unless independent is False; then
			                  they're just zero.
			:param hessians: the diagonals of the hessian with respect to some basis.
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
			self.hessians = values.hessians

		else:
			# ensure the values have enough dimensions
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
				self.gradients = DenseSparseArray.zeros((), ())
			# if hessians are given
			if hessians is not None:
				assert type(hessians) is DenseSparseArray
				if hessians.shape != self.values.shape + 2*self.gradients.shape[self.values.ndim:]:
					raise IndexError("the given array dimensions do not match")
				self.hessians = hessians
			elif self.gradients.ndim > 0:
				self.hessians = DenseSparseArray.zeros(
					self.values.shape, 2*self.gradients.shape[self.values.ndim:])
			else:
				self.hessians = DenseSparseArray.zeros((), ())

		self.shape = self.values.shape
		""" the shape of self.values """
		self.space = self.gradients.shape[self.values.ndim:]
		""" the shape of each gradient """
		self.g_shp = (slice(None),)*len(self.shape) + (np.newaxis,)*len(self.space)
		""" this tuple should be used to index self.values when they need to broadcast
		    to the shape of self.gradients
		"""
		self.c_shp = (slice(None),)*len(self.shape) + (np.newaxis,)*len(self.space)*2
		""" this tuple should be used to index self.values when they need to broadcast
		    to the shape of self.hessians
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
		hessian_index = (slice(None),)*2*len(self.space)
		return Variable(self.values[value_index],
		                self.gradients[(*value_index, *gradient_index)],
		                self.hessians[(*value_index, *hessian_index)])

	def __add__(self, other):
		other = Variable(other)
		return Variable(self.values + other.values,
		                self.gradients + other.gradients,
		                self.hessians + other.hessians)

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
		                gradients=self.gradients*other.values[self.g_shp] +
		                          other.gradients*self.values[self.g_shp],
		                hessians=self.hessians * other.values[self.c_shp] +
		                         self.gradients.outer_multiply(other.gradients) +
		                         other.gradients.outer_multiply(self.gradients) +
		                         other.hessians * self.values[self.c_shp])

	def __neg__(self):
		return self * (-1)

	def __pow__(self, power):
		return Variable(values=self.values**power,
		                gradients=self.gradients*self.values[self.g_shp]**(power - 1)*power,
		                hessians=(self.gradients.outer_multiply(self.gradients)*(power - 1) +
		                          self.hessians*self.values[self.c_shp])*
		                         self.values[self.c_shp]**(power - 2)*power)

	def __sub__(self, other):
		other = Variable(other)
		return Variable(self.values - other.values,
		                self.gradients - other.gradients,
		                self.hessians - other.hessians)

	def __truediv__(self, other):
		return self * other**(-1)

	def __rtruediv__(self, other):
		return other * self**(-1)

	def __radd__(self, other):
		return self + other

	def __rsub__(self, other):
		return -self + other

	def __rmatmul__(self, other):
		return Variable(other@self.values, other@self.gradients, other@self.hessians)

	def __rmul__(self, other): # watch out! never multiply ndarray*Variable, as I can't figure out how to override Numpy's bad behavior there
		return self * other

	def sqrt(self):
		return self ** 0.5

	def log(self):
		return Variable(values=np.log(self.values),
		                gradients=self.gradients/self.values[self.g_shp],
		                hessians=(-self.gradients.outer_multiply(self.gradients)/self.values[self.c_shp] +
		                          self.hessians)/self.values[self.c_shp])

	def exp(self):
		return Variable(values=np.exp(self.values),
		                gradients=self.gradients*np.exp(self.values[self.g_shp]),
		                hessians=(self.gradients.outer_multiply(self.gradients) +
		                          self.hessians)*np.exp(self.values[self.c_shp]))

	def sum(self, axis=None):
		if axis is None:
			axis = tuple(np.arange(self.ndim))
		else:
			axis = np.atleast_1d(axis)
			axis = (axis + self.ndim)%self.ndim
			axis = tuple(axis)
		return Variable(self.values.sum(axis=axis),
		                self.gradients.sum(axis=axis),
		                self.hessians.sum(axis=axis))


if __name__ == "__main__":
	import matplotlib.pyplot as plt
	x0 = Variable(np.linspace(1, 3, 6).reshape((2, 3)), DenseSparseArray.identity((2, 3)))
	d = np.linspace(-2, -3, 6).reshape((2, 3))

	def f(x):
		return (np.log(x)**3).sum()

	f0 = f(x0)
	value = f0.values[()]
	gradient = np.array(f0.gradients)
	hessian = f0.hessians.make_axes_dense(d.ndim)

	steps, values, expectations = [], [], []
	for h in np.linspace(-1, 1):
		steps.append(h)
		values.append(f(x0 + h*d).values[()])
		expectations.append(value + np.sum((h*d)*gradient) + 1/2*np.sum((h*d)*(hessian@(h*d))))
	plt.figure()
	plt.plot(steps, values)
	plt.plot(steps, expectations, "--")
	plt.show()
