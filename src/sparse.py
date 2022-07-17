#!/usr/bin/env python
"""
sparse.py

a custom sparse array class.  it's kind of slow because I implemented it wholly in python,
but it gets the job done.
"""
from __future__ import annotations

import numpy as np


class SparseArray:
	def __init__(self, shape: list[int], indices: np.ndarray = None, values: np.ndarray = None):
		""" an array that takes up less memory than usual.  this implementation is a
		    coordinate array that avoids duplicate indices.
		    :param shape: the number of allowable indices on each axis
		    :param indices: the indices at which there are nonzero elements.  each row
		                    gives the location of one element
		    :param values: the nonzero elements, if there are any.
		"""
		if indices is not None:
			assert indices.dtype == int
		if indices.ndim != 2 or indices.shape[1] != len(shape):
			raise ValueError("you don't seem to understand how indices work")
		if values.ndim != 1:
			raise ValueError("there is only one axis of values in a sparse array")
		self.shape = shape
		self.ndim = len(shape)
		# if indices and values are provided
		if indices is not None:
			assert values is not None # they must both be provided
			# fill up to however many values they have
			self.num_elements = indices.shape[0]
			self.indices = indices
			self.values = values
		# if indices and values are omitted
		else:
			assert values is None # they must both be omitted
			# don't fill it at all (obviusly)
			self.num_elements = 0
			self.indices = -np.empty((0, len(shape)), dtype=int)
			self.values = np.empty(0)

	def __str__(self):
		s = "SparseArray(\n"
		for i in range(self.num_elements):
			s += f"            ({','.join(str(j) for j in self.indices[i, :])}): {self.values[i]:.6g}\n"
		s += ")"
		return s

	def __add__(self, other: SparseArray | np.ndarray | float):
		if type(other) is SparseArray:
			if self.shape != other.shape:
				raise IndexError("these array sizes must match")
			# look for indices that overlap
			corresponding_elements = np.all(np.equal(self.indices[:, np.newaxis, :],
			                                         other.indices[np.newaxis, :, :]), axis=2)
			i_match, j_match = np.nonzero(corresponding_elements)
			# look for indices that are only in one of the two arrays
			i_lone = np.nonzero(~np.any(corresponding_elements, axis=1))[0]
			j_lone = np.nonzero(~np.any(corresponding_elements, axis=0))[0]
			# combine it all
			new_indices = np.concatenate([self.indices[i_match, :], self.indices[i_lone, :], other.indices[j_lone, :]])
			new_values = np.concatenate([self.values[i_match] + other.values[j_match], self.values[i_lone], other.values[j_lone]])
			return SparseArray(self.shape, new_indices, new_values)
		else:
			raise NotImplementedError(f"this wouldn't be sparse anymore...? {type(other)}")

	def __mul__(self, other: SparseArray | np.ndarray | float):
		if type(other) is SparseArray:
			if self.shape != other.shape:
				raise IndexError("these array sizes must match")
			# look for indices that overlap
			corresponding_elements = np.all(np.equal(self.indices[:, np.newaxis, :],
			                                         other.indices[np.newaxis, :, :]), axis=2)
			i_match, j_match = np.nonzero(corresponding_elements)
			# multiply just those ones
			new_indices = self.indices[i_match, :]
			new_values = self.values[i_match] * other.values[j_match]
			return SparseArray(self.shape, new_indices, new_values)
		else:
			return SparseArray(self.shape, self.indices, self.values*other)

	def __truediv__(self, other: np.ndarray | float):
		return SparseArray(self.shape, self.indices, self.values/other)

	@staticmethod
	def find_matching_layer(pattern: np.ndarray, layers: np.ndarray):
		for k in range(layers.shape[-1]):
			if np.array_equal(pattern, layers[..., k]):
				return k
		return layers.shape[-1]

	def __pow__(self, power):
		return SparseArray(self.shape, self.indices, self.values**power)

class DenseSparseArray:
	def __init__(self, shape: list[int], arrays: np.ndarray):
		""" an array that is dense on the first few axes and sparse on the latter few. """
		for array in np.nditer(arrays, flags=["refs_ok"]):
			if shape != arrays.shape + array[()].shape:
				raise IndexError(f"askd for {shape} but it's {arrays.shape} with {array[()].shape}")
		self.shape = shape
		self.ndim = len(shape)
		self.size = np.product(shape)
		self.arrays = arrays

	@staticmethod
	def zeros(dense_shape: tuple | list[int], sparse_shape: tuple | list[int]):
		""" return an array of zeros with some dense axes and some empty sparse axes """
		def create_value(*indices: int):
			return SparseArray(sparse_shape,
			                   np.empty((0, len(sparse_shape)), dtype=int),
			                   np.empty((0,)))
		create_values = np.vectorize(create_value)
		return DenseSparseArray(dense_shape + sparse_shape,
		                        create_values(*np.indices(dense_shape)))

	@staticmethod
	def identity(shape: tuple | list[int]):
		""" return an array where the dense part of the shape is the same as the sparse
		    part of the shape, and each sparse array contains a single 1 in the pasition
		    corresponding to its own position in the dense array.  this is the output
		    you would expect from identity(product(shape)).reshape((*shape, *shape)), but
		    I don't want to implement reshape.
		"""
		def create_value(*indices: int):
			return SparseArray(shape, np.array([indices]), np.array([1]))
		create_values = np.vectorize(create_value)
		return DenseSparseArray(shape*2,
		                        create_values(*np.indices(shape)))

	def __str__(self):
		dense = np.zeros(self.shape)
		array_iterator = np.nditer(self.arrays, flags=['multi_index', 'refs_ok'])
		for array in array_iterator:
			array = array[()]
			dense[array_iterator.multi_index + tuple(array.indices.T)] = array.values
		return str(dense)

	def __getitem__(self, item):
		if type(item) is not tuple:
			raise NotImplementedError("this index is too complicated")
		# start by expanding any ...
		for i in range(len(item)):
			if item[i] is Ellipsis:
				item = item[:i] + (slice(None),)*(len(self.shape) - len(item) - 1) + item[i + 1:]
		# then check that it's safe to chop off the sparse part of the index
		for i in range(self.arrays.ndim, self.ndim):
			if item[i] != slice(None):
				raise NotImplementedError("this SparseArray implementation does not support indexing on the sparse axis")
		# finally, apply the requested slice or whatever to the main array
		value = self.arrays[item[:self.arrays.ndim]]
		return DenseSparseArray(value.shape + self.shape[self.arrays.ndim:], value)

	def __add__(self, other: DenseSparseArray | np.ndarray | float):
		try:
			return DenseSparseArray(self.shape, self.arrays + other.arrays)
		except AttributeError:
			if np.all(np.equal(other, 0)):
				return self
			else:
				raise TypeError("DenseSparseArrays cannot be added to normal arrays (unless the normal array is just 0)")

	def __sub__(self, other: DenseSparseArray | np.ndarray | float):
		return self + other * -1

	def __mul__(self, other: DenseSparseArray | np.ndarray | float):
		try:
			return DenseSparseArray(self.shape, self.arrays * other.arrays)
		except AttributeError:
			try:
				if self.ndim == other.ndim:
					if np.all(np.equal(other.shape[len(self.arrays.shape):], 1)):
						sparse_axes = tuple(np.arange(self.arrays.ndim, self.ndim))
						return DenseSparseArray(
							self.shape, self.arrays * np.squeeze(other, axis=sparse_axes))
					else:
						raise NotImplementedError("I don't support elementwise operations on the sparse axes")
				elif other.ndim == 0:
					return DenseSparseArray(self.shape, self.arrays * other[()])
				else:
					raise IndexError(f"array shapes do not match: {self.shape} and {other.shape}")
			except AttributeError:
				return DenseSparseArray(self.shape, self.arrays * other)

	def __truediv__(self, other: np.ndarray | float):
		return self * (1/other)

	def __pow__(self, power: float):
		return DenseSparseArray(self.shape, self.arrays ** power)

	def sum(self, axis):
		# if summing over all dense axes, convert to a regular dense array
		if len(axis) >= len(self.arrays.shape):
			# there are a lot of generalizations I could make here but don't want to, so I assert them away
			assert len(axis) == len(self.arrays.shape)
			assert np.all(np.less(axis, self.arrays.ndim))
			dense = np.zeros(self.shape[self.arrays.ndim:])
			for sparse in np.nditer(self.arrays, flags=['refs_ok']):
				sparse = sparse[()]
				dense[tuple(sparse.indices.T)] += sparse.values
			return dense
		# if summing over one axis, numpy will hopefully do this for me
		elif len(axis) == 1:
			if axis[0] >= self.arrays.ndim or axis[0] < 0:
				raise IndexError("I haven't implemented summing on the sparse axes.")
			arrays = np.sum(self.arrays, axis=axis)
			return DenseSparseArray(arrays.shape + tuple(self.shape[self.arrays.ndim:]),
			                        arrays)
		else:
			raise NotImplementedError("I haven't implemented... whatever it is you're trying to do.")


if __name__ == "__main__":
	array = DenseSparseArray(
		[1, 3, 10],
		np.array([[SparseArray([10],
		                       np.array([[0], [1]]),
		                       np.array([1., -1.])),
		           SparseArray([10],
		                       np.array([[4]]),
		                       np.array([1.])),
		           SparseArray([10],
		                       np.array([[5], [0]]),
		                       np.array([-2., 3.])),
		]])
	)
	brray = DenseSparseArray(
		[1, 3, 10],
		np.array([[SparseArray([10],
		                       np.array([[1], [8]]),
		                       np.array([1., 0.])),
		           SparseArray([10],
		                       np.array([[4], [0]]),
		                       np.array([3., -2.])),
		           SparseArray([10],
		                       np.array([[6]]),
		                       np.array([-2.])),
		]])
	)
	print(array)
	print(brray)
	print(array + brray)
	print(array*brray)
	print(array**2)
	print(array.sum(axis=(0,)))
	print(array.sum(axis=(1,)))
	print(array.sum(axis=(0, 1)))
	print(array.sum(axis=(0,)).sum(axis=(0,)))
