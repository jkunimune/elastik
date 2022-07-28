#!/usr/bin/env python
"""
sparse.py

a custom sparse array class.  it dips into the shadow realm to make the big calculations faster.
"""
from __future__ import annotations

import os
import sys
from ctypes import c_int, c_void_p, Structure, cdll, CDLL, Array, POINTER, c_double, cast
from typing import Callable, Sequence, Collection

import numpy as np


if sys.platform.startswith('win32'):
	c_lib = cdll.LoadLibrary("../lib/libsparse.dll")
elif sys.platform.startswith('linux'):
	c_lib = CDLL(os.path.join(os.getcwd(), "../lib/libsparse.so.1.0.0"))
else:
	raise OSError(f"I don't recognize the platform {sys.platform}")


c_int_p = POINTER(c_int)
c_ndarray = np.ctypeslib.ndpointer(dtype=c_double, flags='C_CONTIGUOUS')

class c_SparseArrayArray(Structure):
	_fields_ = [("ndim", c_int),
	            ("size", c_int),
	            ("shape", c_int_p),
	            ("elements", c_void_p),
	            ]

def c_int_array(lst: Sequence[int]) -> Array:
	return (c_int*len(lst))(*lst)

def ndarray_from_c(data: c_ndarray, shape: Sequence[int]) -> np.ndarray:
	return np.ctypeslib.as_array(cast(data, POINTER(c_double)), shape=shape) # TODO: someone on the internet sed that Python won't free this memory even tho the documentation ses that it shares memory with the ctypes input and the git repo ses that tit used to leak but has ben fixd... so check this for leaks

def declare_c_func(func: Callable, args: list[type], res: type | None):
	func.argtypes = args
	func.restype = res

# declare all of the C functions we plan to use and their parameters' types
declare_c_func(c_lib.free_saa, [c_SparseArrayArray], None)
declare_c_func(c_lib.add_saa, [c_SparseArrayArray, c_SparseArrayArray], c_SparseArrayArray)
declare_c_func(c_lib.subtract_saa, [c_SparseArrayArray, c_SparseArrayArray], c_SparseArrayArray)
declare_c_func(c_lib.multiply_saa, [c_SparseArrayArray, c_SparseArrayArray], c_SparseArrayArray)
declare_c_func(c_lib.zeros, [c_int, c_int_p, c_int], c_SparseArrayArray)
declare_c_func(c_lib.identity, [c_int, c_int_p], c_SparseArrayArray)
declare_c_func(c_lib.unit, [c_int, c_int_p, c_int_p, c_int, c_int_p, c_double], c_SparseArrayArray)
declare_c_func(c_lib.multiply_nda, [c_SparseArrayArray, c_ndarray, c_int_p], c_SparseArrayArray)
declare_c_func(c_lib.divide_nda, [c_SparseArrayArray, c_ndarray, c_int_p], c_SparseArrayArray)
declare_c_func(c_lib.multiply_f, [c_SparseArrayArray, c_double], c_SparseArrayArray)
declare_c_func(c_lib.divide_f, [c_SparseArrayArray, c_double], c_SparseArrayArray)
declare_c_func(c_lib.power_f, [c_SparseArrayArray, c_double], c_SparseArrayArray)
declare_c_func(c_lib.sum_along_axis, [c_SparseArrayArray, c_int], c_SparseArrayArray)
declare_c_func(c_lib.sum_all, [c_SparseArrayArray, c_int_p], c_ndarray)
declare_c_func(c_lib.to_dense, [c_SparseArrayArray, c_int_p], c_ndarray)
declare_c_func(c_lib.get_slice_saa, [c_SparseArrayArray, c_int, c_int], c_SparseArrayArray)
declare_c_func(c_lib.get_reindex_saa, [c_SparseArrayArray, c_int_p, c_int, c_int], c_SparseArrayArray)


class DenseSparseArray:
	def __init__(self, shape: Sequence[int], c_struct: c_SparseArrayArray):
		""" an array that is dense on the first few axes and sparse on the others. """
		self.shape = tuple(shape)
		self.ndim = len(shape)
		self.size = np.product(shape)
		self.dense_ndim = c_struct.ndim
		self.dense_shape = shape[:self.dense_ndim]
		self.sparse_ndim = c_struct.ndim
		self.sparse_shape = shape[self.dense_ndim:]
		self._as_parameter_ = c_struct

	def __del__(self):
		c_lib.free_saa(self)

	@staticmethod
	def zeros(dense_shape: Sequence[int], sparse_shape: Sequence[int]):
		""" return an array of zeros with some dense axes and some empty sparse axes """
		return DenseSparseArray(tuple(dense_shape) + tuple(sparse_shape),
		                        c_lib.zeros(c_int(len(dense_shape)),
		                                    c_int_array(dense_shape),
		                                    c_int(len(sparse_shape))))

	@staticmethod
	def identity(shape: Sequence[int]):
		""" return an array where the dense part of the shape is the same as the sparse
		    part of the shape, and each sparse array contains a single 1 in the pasition
		    corresponding to its own position in the dense array.  this is the output
		    you would expect from identity(product(shape)).reshape((*shape, *shape)), but
		    I don't want to implement reshape.
		"""
		return DenseSparseArray(tuple(shape)*2, c_lib.identity(c_int(len(shape)), c_int_array(shape)))

	@staticmethod
	def unit(dense_shape: Sequence[int], dense_index: Sequence[int],
	         sparse_shape: Sequence[int], sparse_index: Sequence[int], value: float):
		""" return an array with a single nonzero element.  mostly for testing purposes. """
		return DenseSparseArray(tuple(dense_shape) + tuple(sparse_shape),
		                        c_lib.unit(c_int(len(dense_shape)),
		                                   c_int_array(dense_shape),
		                                   c_int_array(dense_index),
		                                   c_int(len(sparse_shape)),
		                                   c_int_array(sparse_index),
		                                   c_double(value)))

	def __add__(self, other: DenseSparseArray | np.ndarray | float) -> DenseSparseArray:
		if type(other) is DenseSparseArray:
			return DenseSparseArray(self.shape, c_lib.add_saa(self, other))
		else:
			if np.all(np.equal(other, 0)):
				return self
			else:
				raise TypeError("DenseSparseArrays cannot be added to normal arrays (unless the normal array is just 0)")

	def __sub__(self, other: DenseSparseArray | np.ndarray | float) -> DenseSparseArray:
		if type(other) is DenseSparseArray:
			return DenseSparseArray(self.shape, c_lib.subtract_saa(self, other))
		else:
			if np.all(np.equal(other, 0)):
				return self
			else:
				raise TypeError("DenseSparseArrays cannot be subtracted from normal arrays (unless the normal array is just 0)")

	def __mul__(self, other: DenseSparseArray | np.ndarray | float):
		other = self.convert_arg_for_c(other)
		if type(other) is DenseSparseArray:
			return DenseSparseArray(self.shape, c_lib.multiply_saa(self, other))
		if type(other) is np.ndarray:
			return DenseSparseArray(self.shape, c_lib.multiply_nda(self, other, c_int_array(other.shape)))
		elif type(other) is c_double:
			return DenseSparseArray(self.shape, c_lib.multiply_f(self, other))
		else:
			raise TypeError(f"can't multiply by {other}")

	def __truediv__(self, other: np.ndarray | float):
		other = self.convert_arg_for_c(other)
		if type(other) is np.ndarray:
			return DenseSparseArray(self.shape, c_lib.divide_nda(self, other, c_int_array(other.shape)))
		elif type(other) is c_double:
			return DenseSparseArray(self.shape, c_lib.divide_f(self, other))
		else:
			raise TypeError(f"can't divide by {type(other)}")

	def __pow__(self, power: float):
		return DenseSparseArray(self.shape, c_lib.power_f(self, c_double(power)))

	def convert_arg_for_c(self, other: DenseSparseArray | np.ndarray | float) -> DenseSparseArray | np.ndarray | c_double:
		""" convert the given argument to a form in which it is compatible with c, and
		    also compatible with self._as_attribute_ in terms of ndim and shape and all
		    that, so that they can reasonbaly operate with each other in the shadow realm
		"""
		if type(other) is DenseSparseArray:
			if self.shape == other.shape and self.dense_shape == other.dense_shape:
				return other
			else:
				raise IndexError(f"array shapes do not match: {self.shape} and {other.shape}")
		elif type(other) is np.ndarray:
			if self.ndim == other.ndim:
				if np.all(np.equal(other.shape, self.shape) | np.equal(other.shape, 1)):
					if np.all(np.equal(other.shape[self.dense_ndim:], 1)):
						sparse_axes = tuple(np.arange(self.dense_ndim, self.ndim))
						return np.squeeze(other, axis=sparse_axes).astype(float)
					else:
						raise NotImplementedError("I don't support elementwise operations on the sparse axes")
				else:
					raise IndexError(f"array shapes do not match: {self.shape} and {other.shape}")
			elif other.ndim == 0:
				return c_double(other[()])
			else:
				raise IndexError(f"array shapes do not match: {self.shape} and {other.shape}")
		elif type(other) is int or type(other) is float:
			return c_double(other)
		else:
			raise TypeError(f"can't multiply by {other!r}")

	def __getitem__(self, index: tuple) -> DenseSparseArray:
		if type(index) is not tuple:
			raise NotImplementedError(f"this index is too complicated, {index!r}")
		# start by expanding any ...
		for i in range(len(index)):
			if index[i] is Ellipsis:
				index = index[:i] + (slice(None),)*(len(self.shape) - len(index) - 1) + index[i + 1:]
		if len(index) != self.ndim:
			raise IndexError(f"this index has {len(index)} indices but we can only index {self.ndim}")

		# then go thru and do each item one at a time
		result = self._as_parameter_
		shape = []
		for k in range(self.ndim - 1, -1, -1):
			if type(index[k]) is slice and index[k] == slice(None):
				shape.append(self.shape[k])
			elif k >= self.dense_ndim:
				raise NotImplementedError(f"this SparseArray implementation does not support indexing on the sparse axis: {index}")
			elif type(index[k]) == np.ndarray:
				if index[k].ndim == 1:
					result = c_lib.get_reindex_saa(result, c_int_array(index[k]), c_int(index[k].size), c_int(k))
					shape.append(index[k].size)
				else:
					raise NotImplementedError("only 1D numpy ndarrays may be used to index")
			elif type(index[k]) == int:
				result = c_lib.get_slice_saa(result, c_int(index[k]), c_int(k))
			else:
				raise NotImplementedError(f"I can't do this index, {index[k]!r}")
		return DenseSparseArray(shape[::-1], result)

	def __str__(self):
		return str(np.array(self))

	def __array__(self):
		print("look out: it's converting to a dense array")
		return ndarray_from_c(c_lib.to_dense(self, c_int_array(self.sparse_shape)), self.shape)

	def sum(self, axis: Collection[int] | int | None) -> DenseSparseArray | np.ndarray:
		if type(axis) is int:
			axis = [axis]
		if axis is None or np.any(np.greater_equal(axis, self.dense_ndim)):
			raise NotImplementedError("I haven't implemented summing on the sparse axes")
		elif np.any(np.less(axis, 0)):
			raise ValueError("all of the axes must be nonnegative (sorry)")
		unique, instances = np.unique(axis, return_counts=True)
		if np.any(np.greater(instances, 1)):
			raise ValueError("the axes can't have duplicates")
		# if summing over all dense axes, convert to a regular dense array
		if len(axis) == self.dense_ndim:
			return ndarray_from_c(c_lib.sum_all(self, c_int_array(self.sparse_shape)), self.sparse_shape)
		# otherwise, do them one at a time
		else:
			result = self._as_parameter_
			shape = list(self.shape[:])
			for k in sorted(axis, reverse=True):
				result = c_lib.sum_along_axis(result, c_int(k))
				shape.pop(k)
			return DenseSparseArray(shape, result)


if __name__ == "__main__":
	array = DenseSparseArray.zeros([1, 3], [10])
	array += DenseSparseArray.unit([1, 3], [0, 0], [10], [0], 1.)
	array += DenseSparseArray.unit([1, 3], [0, 0], [10], [1], -1.)
	array += DenseSparseArray.unit([1, 3], [0, 1], [10], [4], 1.)
	array += DenseSparseArray.unit([1, 3], [0, 2], [10], [5], -2.)
	array += DenseSparseArray.unit([1, 3], [0, 2], [10], [0], 3.)

	brray = DenseSparseArray.zeros([1, 3], [10])
	brray += DenseSparseArray.unit([1, 3], [0, 0], [10], [1], 1.)
	brray += DenseSparseArray.unit([1, 3], [0, 0], [10], [0], 0.)
	brray += DenseSparseArray.unit([1, 3], [0, 1], [10], [4], 3.)
	brray += DenseSparseArray.unit([1, 3], [0, 1], [10], [0], -2.)
	brray += DenseSparseArray.unit([1, 3], [0, 2], [10], [6], -2.)
	brray += DenseSparseArray.unit([1, 3], [0, 2], [10], [5], -2.)

	# brray = np.array([-3, 2, 0]).reshape((1, 3, 1))

	# brray = np.array([[[-3]]])

	print("A =", array)
	print("B =", brray)
	print("A + B =", array + brray)
	print("A - B =", array - brray)
	print("A*B =", array*brray)
	# print("A/B =", array/brray)
	print("A*2 =", array*2)
	print("A/2 =", array/2)
	print("A^2 =", array**2)
	print(array[0, :, :])
	print(array[:, 1, :])
	print(array[:, np.array([2, 1, 0]), :])
	print(array[0, np.array([2, 1, 0]), :])
	print(array.sum(axis=[0]))
	print(array.sum(axis=1))
	print(array.sum(axis=[0, 1]))
	print(array.sum(axis=0).sum(axis=[0]))

	# eyes = DenseSparseArray.identity(([4, 2]))
	# print(eyes)
