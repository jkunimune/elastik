#!/usr/bin/env python
"""
autodiff.py

a file that defines a class that can be used in place of a numpy float array, but that will
automaticly calculate its own gradients and hessians when arithmetic operations are applied to it.
very useful for gradient- and hessian-based optimization rootenes.
"""
from __future__ import annotations

from functools import cached_property
from typing import Sequence, Optional

import numpy as np
from numpy.typing import NDArray

from sparse import SparseNDArray


class Variable:
	def __init__(self,
	             values: NDArray[float],
	             gradients: Optional[SparseNDArray],
	             hessians: Optional[SparseNDArray]):
		""" an array of values with gradient information attached, for computing gradients and
		    hessians of vectorized functions
			:param values: the local value of the quantity
			:param gradients: the gradients of the values with respect to some basis
			:param hessians: the diagonals of the hessian with respect to some basis
		"""
		# handle the dimensions stuff
		self.shape = tuple(values.shape)
		self.size = np.product(self.shape, dtype=int)
		self.ndim = len(self.shape)
		self.domain_shape = gradients.shape[self.ndim:]
		self.domain_size = np.product(self.domain_shape, dtype=int)
		self.domain_ndim = len(self.domain_shape)

		# ensure the values have the rite size
		if values.shape != self.shape:
			raise ValueError(f"the given values are not shaped rite (you passed {values.shape} but we need {self.size}")
		self.values = values

		if type(gradients) is not SparseNDArray:
			raise TypeError("this really does need to be a sparse matrix")
		if gradients.shape != self.shape + self.domain_shape:
			raise ValueError(f"the given array dimensions do not match (you passd {gradients.shape} but "
			                 f"we need {(self.size, self.domain_size)}).")
		self.gradients = gradients

		if type(hessians) is not SparseNDArray:
			raise TypeError("this really does need to be a sparse matrix")
		if hessians.shape != self.shape + 2*self.domain_shape:
			raise ValueError("the given hessian has the rong shape")
		self.hessians = hessians

	@staticmethod
	def convert(other: NDArray[float] | Variable, shape: Sequence[int], domain_shape: Sequence[int]) -> Variable:
		if type(other) == Variable:
			return other
		else:
			return Variable.create_constant(np.broadcast_to(other, shape), domain_shape)

	@staticmethod
	def create_constant(values: NDArray[float] | float, domain_shape: Sequence[int]) -> Variable:
		""" create a Variable with zero derivative or gradient """
		values = np.array(values)
		return Variable(
			values,
			SparseNDArray.zeros(values.shape + tuple(domain_shape), len(domain_shape)),
			SparseNDArray.zeros(values.shape + 2*tuple(domain_shape), 2*len(domain_shape)))

	@staticmethod
	def create_independent(values: NDArray[float]) -> Variable:
		""" create a Variable out of some given values that will form the input to a function,
		    and the function’s eventual output will have a gradient that tracks its change in terms
		    of these values here
		"""
		return Variable(values,
		                SparseNDArray.identity(values.shape),
		                SparseNDArray.zeros(3*values.shape, 2*values.ndim))

	@staticmethod
	def create_scan(values: NDArray[float]) -> Variable:
		""" create a Variable out of some given values with a single item in the gradient, such that
		    the gradient is really the derivative with respect to the scalar input evaluated at
		    multiple values
		"""
		return Variable(values,
		                SparseNDArray.from_dense(np.ones(values.shape), 0),
		                SparseNDArray.zeros(values.shape, 0))

	@cached_property
	def value(self) -> float:
		""" the value of this scalar Variable """
		if self.ndim == 0:
			return self.values[()]
		else:
			raise ValueError("the value attribute is only to be used for scalar Variables")

	@cached_property
	def gradient(self) -> NDArray[float]:
		""" the gradient of this scalar Variable as a dense array """
		if self.ndim == 0:
			return self.gradients.__array__()
		else:
			raise ValueError("the gradient attribute is only to be used for scalar Variables")

	@cached_property
	def hessian(self) -> SparseNDArray:
		""" the hessian of this scalar Variable as a 2D sparse array """
		if self.ndim == 0:
			return self.hessians.reshape(self.domain_shape*2, self.domain_ndim)
		else:
			raise ValueError("the hessian attribute is only to be used for scalar Variables")

	def __str__(self) -> str:
		return f"{'x'.join(str(i) for i in self.shape)}({'x'.join(str(i) for i in self.domain_shape)})"

	def __getitem__(self, item: tuple) -> Variable:
		domain_slice = (slice(None),)*self.domain_ndim
		values = self.values[item]
		gradients = self.gradients[item + domain_slice]
		hessians = self.hessians[item + 2*domain_slice]
		return Variable(values, gradients, hessians)

	def __add__(self, other: Variable | NDArray[float] | float) -> Variable:
		if type(other) is not Variable:
			if np.ndim(other) > self.ndim or np.any(np.greater(np.shape(other), self.shape[:np.ndim(other)])):
				raise ValueError("I haven’t implemented up-broadcasting of Variables")
			return Variable(self.values + other, self.gradients, self.hessians)
		else:
			if other.shape != self.shape:
				raise ValueError(f"the shapes don’t match: {self.shape} and {other.shape}")
			other = Variable.convert(other, self.shape, self.domain_shape)
			return Variable(self.values + other.values,
			                self.gradients + other.gradients,
			                self.hessians + other.hessians)

	def __le__(self, other: Variable | NDArray[float] | float) -> NDArray[bool]:
		other = Variable.convert(other, self.shape, self.domain_shape)
		return self.values <= other.values

	def __lt__(self, other: Variable | NDArray[float] | float) -> NDArray[bool]:
		other = Variable.convert(other, self.shape, self.domain_shape)
		return self.values < other.values

	def __ge__(self, other: Variable | NDArray[float] | float) -> NDArray[bool]:
		other = Variable.convert(other, self.shape, self.domain_shape)
		return self.values >= other.values

	def __gt__(self, other: Variable | NDArray[float] | float) -> NDArray[bool]:
		other = Variable.convert(other, self.shape, self.domain_shape)
		return self.values > other.values

	def __mul__(self, other: Variable | NDArray[float] | float) -> Variable:
		if type(other) is not Variable and np.ndim(other) == 0:
			return Variable(self.values*other, self.gradients*other, self.hessians*other)
		else:
			other = Variable.convert(other, self.shape, self.domain_shape)
			grad_slice = (...,) + self.domain_ndim*(np.newaxis,)
			hess_slice = (...,) + 2*self.domain_ndim*(np.newaxis,)
			return Variable(values=self.values * other.values,
			                gradients=self.gradients * other.values[grad_slice] +
			                          other.gradients * self.values[grad_slice],
			                hessians=self.hessians * other.values[hess_slice] +
			                         self.gradients.outer_multiply(other.gradients) +
			                         other.gradients.outer_multiply(self.gradients) +
			                         other.hessians * self.values[hess_slice])

	def __neg__(self) -> Variable:
		return self * (-1)

	def __pow__(self, power: float) -> Variable:
		gradients_sqr = self.gradients.outer_multiply(self.gradients)
		grad_slice = (...,) + self.domain_ndim*(np.newaxis,)
		hess_slice = (...,) + 2*self.domain_ndim*(np.newaxis,)
		return Variable(values=self.values**power,
		                gradients=self.gradients*self.values[grad_slice]**(power - 1)*power,
		                hessians=(gradients_sqr*(power - 1) +
		                          self.hessians*self.values[hess_slice])*
		                          self.values[hess_slice]**(power - 2)*power)

	def __sub__(self, other: Variable | NDArray[float] | float) -> Variable:
		return self + (-other)

	def __truediv__(self, other: Variable | NDArray[float] | float) -> Variable:
		return self * other**(-1)

	def __rtruediv__(self, other: Variable | NDArray[float] | float) -> Variable:
		return other * self**(-1)

	def __radd__(self, other: Variable | NDArray[float] | float) -> Variable:
		return self + other

	def __rsub__(self, other: Variable | NDArray[float] | float) -> Variable:
		return -self + other

	def __rmatmul__(self, other: SparseNDArray) -> Variable:
		if type(other) is not SparseNDArray:
			return NotImplemented
		return Variable(other@self.values,
		                other@self.gradients,
		                other@self.hessians)

	def __rmul__(self, other: Variable | NDArray[float] | float) -> Variable: # watch out! never multiply ndarray*Variable, as I can't figure out how to override Numpy's bad behavior there
		return self * other

	def sqrt(self) -> Variable:
		return self ** (1/2)

	def log(self) -> Variable:
		gradients_sqr = self.gradients.outer_multiply(self.gradients)
		grad_slice = (...,) + self.domain_ndim*(np.newaxis,)
		hess_slice = (...,) + 2*self.domain_ndim*(np.newaxis,)
		return Variable(values=np.log(self.values),
		                gradients=self.gradients/self.values[grad_slice],
		                hessians=(self.hessians - gradients_sqr/self.values[hess_slice])/
		                         self.values[hess_slice])

	def exp(self) -> Variable:
		exp_values = np.exp(self.values)
		gradients_sqr = self.gradients.outer_multiply(self.gradients)
		grad_slice = (...,) + self.domain_ndim*(np.newaxis,)
		hess_slice = (...,) + 2*self.domain_ndim*(np.newaxis,)
		return Variable(exp_values,
		                self.gradients*exp_values[grad_slice],
		                (self.hessians + gradients_sqr)*exp_values[hess_slice])

	def sum(self) -> Variable:
		dense_axes = np.arange(self.ndim)
		return Variable(self.values.sum(),
		                self.gradients.sum(axis=dense_axes),
		                self.hessians.sum(axis=dense_axes))


def test():
	import matplotlib.pyplot as plt
	x0 = Variable.create_independent(np.linspace(1, 3, 6).reshape((2, 3)))
	d = np.linspace(-2, -3, 6).reshape((2, 3))

	def f(x):
		y = x[0, :] - x[1, :] + 6.
		z = np.sqrt(3*x[0, :]**2 + x[1, :]**2)
		return (np.log(y*z)**3).sum()

	f0 = f(x0)
	value = f0.values[()]
	gradient = np.array(f0.gradient)
	hessian = f0.hessian
	print(hessian)

	steps, values, expectations = [], [], []
	for h in np.linspace(-1, 1):
		steps.append(h)
		values.append(f(x0 + h*d).values[()])
		expectations.append(value + np.sum((h*d)*gradient) + 1/2*np.sum((h*d).ravel()*(hessian.reshape((6,6), 1)@(h*d).ravel())))
	plt.figure()
	plt.plot(steps, values)
	plt.plot(steps, expectations, "--")
	plt.show()


if __name__ == "__main__":
	test()
