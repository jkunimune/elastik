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
c_int_ndarray = np.ctypeslib.ndpointer(dtype=c_int, flags='C_CONTIGUOUS')

class c_SparseArrayArray(Structure):
	_fields_ = [("ndim", c_int),
	            ("size", c_int),
	            ("shape", c_int_p),
	            ("elements", c_void_p),
	            ]

def c_int_array(lst: Sequence[int]) -> Array:
	return (c_int*len(lst))(*lst)

def ndarray_from_c(data: c_ndarray, shape: Sequence[int]) -> np.ndarray:
	c_array = np.ctypeslib.as_array(cast(data, POINTER(c_double)), shape=shape) # bild a np.ndarray around the existing memory
	py_array = c_array.copy() # transcribe it to a python-ownd block of memory
	c_lib.free_nda(c_array) # then delete the old memory from c, since Python won't do it for us
	return py_array

def declare_c_func(func: Callable, args: list[type], res: type | None):
	func.argtypes = args
	func.restype = res

# declare all of the C functions we plan to use and their parameters' types
declare_c_func(c_lib.free_saa, [c_SparseArrayArray], None)
declare_c_func(c_lib.free_nda, [c_ndarray], None)
declare_c_func(c_lib.add_saa, [c_SparseArrayArray, c_SparseArrayArray], c_SparseArrayArray)
declare_c_func(c_lib.subtract_saa, [c_SparseArrayArray, c_SparseArrayArray], c_SparseArrayArray)
declare_c_func(c_lib.multiply_saa, [c_SparseArrayArray, c_SparseArrayArray], c_SparseArrayArray)
declare_c_func(c_lib.matmul_saa, [c_SparseArrayArray, c_SparseArrayArray], c_SparseArrayArray)
declare_c_func(c_lib.zeros, [c_int, c_int_p, c_int], c_SparseArrayArray)
declare_c_func(c_lib.identity, [c_int, c_int_p], c_SparseArrayArray)
declare_c_func(c_lib.new_saa, [c_int, c_int_p, c_int, c_int, c_int_ndarray, c_ndarray], c_SparseArrayArray)
declare_c_func(c_lib.multiply_nda, [c_SparseArrayArray, c_ndarray, c_int_p], c_SparseArrayArray)
declare_c_func(c_lib.divide_nda, [c_SparseArrayArray, c_ndarray, c_int_p], c_SparseArrayArray)
declare_c_func(c_lib.matmul_nda, [c_SparseArrayArray, c_ndarray, c_int_p, c_int], c_ndarray)
declare_c_func(c_lib.multiply_f, [c_SparseArrayArray, c_double], c_SparseArrayArray)
declare_c_func(c_lib.divide_f, [c_SparseArrayArray, c_double], c_SparseArrayArray)
declare_c_func(c_lib.power_f, [c_SparseArrayArray, c_double], c_SparseArrayArray)
declare_c_func(c_lib.sum_along_axis, [c_SparseArrayArray, c_int], c_SparseArrayArray)
declare_c_func(c_lib.sum_all_sparse, [c_SparseArrayArray], c_ndarray)
declare_c_func(c_lib.sum_all_dense, [c_SparseArrayArray, c_int_p], c_ndarray)
declare_c_func(c_lib.to_dense, [c_SparseArrayArray, c_int_p], c_ndarray)
declare_c_func(c_lib.expand_dims, [c_SparseArrayArray, c_int], c_SparseArrayArray)
declare_c_func(c_lib.get_slice_saa, [c_SparseArrayArray, c_int, c_int], c_SparseArrayArray)
declare_c_func(c_lib.get_reindex_saa, [c_SparseArrayArray, c_int_p, c_int, c_int], c_SparseArrayArray)


class DenseSparseArray:
	def __init__(self, shape: Sequence[int], c_struct: c_SparseArrayArray):
		""" an array that is dense on the first few axes and sparse on the others. """
		self.shape = tuple(shape)
		self.ndim = len(shape)
		self.size = np.product(shape)
		self.dense_ndim = c_struct.ndim
		self.dense_shape = self.shape[:self.dense_ndim]
		self.sparse_ndim = self.ndim - self.dense_ndim
		self.sparse_shape = self.shape[self.dense_ndim:]
		self._as_parameter_ = c_struct

	def __del__(self):
		c_lib.free_saa(self)

	@staticmethod
	def zeros(dense_shape: Sequence[int], sparse_shape: Sequence[int]) -> DenseSparseArray:
		""" return an array of zeros with some dense axes and some empty sparse axes """
		return DenseSparseArray(tuple(dense_shape) + tuple(sparse_shape),
		                        c_lib.zeros(c_int(len(dense_shape)),
		                                    c_int_array(dense_shape),
		                                    c_int(len(sparse_shape))))

	@staticmethod
	def identity(shape: int | Sequence[int]) -> DenseSparseArray:
		""" return an array where the dense part of the shape is the same as the sparse
		    part of the shape, and each sparse array contains a single 1 in the position
		    corresponding to its own position in the dense array.  this is the output
		    you would expect from identity(product(shape)).reshape((*shape, *shape)), but
		    I don't want to implement reshape.
		"""
		try:
			shape = tuple(shape)
		except TypeError:
			shape = (shape,)
		return DenseSparseArray(tuple(shape)*2, c_lib.identity(c_int(len(shape)), c_int_array(shape)))

	@staticmethod
	def from_coordinates(sparse_shape: Sequence[int], indices: np.ndarray, values: np.ndarray) -> DenseSparseArray:
		""" return an array where each SparseArray has the same number of nonzero values,
		    and they are located at explicitly known indices
		    :param sparse_shape: the shapes of the SparseArrays this contains
		    :param indices: the indices of the nonzero elements.  this should have shape
		                    (...n)?×m×k, where m is the number of elements in each
		                    SparseArray and k is the number of sparse dimensions. the
		                    shape up to that point corresponds to the dense shape.  these
		                    indices must be sorted and contain no duplicates if the
		                    DenseSparseArray is to operate elementwise on other
		                    DenseSparseArrays.  if you only want to operate on np.ndarrays
		                    or do matrix multiplication, they don't need to be sorted and
		                    duplicates are fine, but I won't sort them for you.  but watch
		                    out for that.
		    :param values: the values of the nonzero elements.  each value corresponds to
		                   one row of indices.
		"""
		if indices.shape[:-1] != values.shape or indices.shape[-1] != len(sparse_shape):
			raise ValueError("you gave the rong number of indices")
		return DenseSparseArray(indices.shape[:-2] + tuple(sparse_shape),
		                        c_lib.new_saa(c_int(indices.ndim - 2),
		                                      c_int_array(indices.shape[:-2]),
		                                      c_int(indices.shape[-2]),
		                                      c_int(indices.shape[-1]),
		                                      indices,
		                                      values))

	def __add__(self, other: DenseSparseArray | np.ndarray | float) -> DenseSparseArray:
		if type(other) is DenseSparseArray:
			if self.dense_shape == other.dense_shape and self.sparse_shape == other.sparse_shape:
				return DenseSparseArray(self.shape, c_lib.add_saa(self, other))
			else:
				raise ValueError("these array sizes do not match")
		elif type(other) is np.ndarray:
			if np.all(np.equal(other, 0)):
				return self
			else:
				raise TypeError("DenseSparseArrays cannot be added to normal arrays (unless the normal array is just 0)")
		else:
			return NotImplemented

	def __sub__(self, other: DenseSparseArray | np.ndarray | float) -> DenseSparseArray:
		if type(other) is DenseSparseArray:
			if self.dense_shape == other.dense_shape and self.sparse_shape == other.sparse_shape:
				return DenseSparseArray(self.shape, c_lib.subtract_saa(self, other))
			else:
				raise ValueError("these array sizes do not match")
		elif type(other) is np.ndarray:
			if np.all(np.equal(other, 0)):
				return self
			else:
				raise TypeError("DenseSparseArrays cannot be subtracted from normal arrays (unless the normal array is just 0)")
		else:
			return NotImplemented

	def __mul__(self, other: DenseSparseArray | np.ndarray | float) -> DenseSparseArray:
		other = self.convert_arg_for_c(other)
		if type(other) is DenseSparseArray:
			return DenseSparseArray(self.shape, c_lib.multiply_saa(self, other))
		if type(other) is np.ndarray:
			return DenseSparseArray(self.shape, c_lib.multiply_nda(self, other, c_int_array(other.shape)))
		elif type(other) is c_double:
			return DenseSparseArray(self.shape, c_lib.multiply_f(self, other))
		else:
			return NotImplemented

	def __truediv__(self, other: np.ndarray | float) -> DenseSparseArray:
		other = self.convert_arg_for_c(other)
		if type(other) is np.ndarray:
			return DenseSparseArray(self.shape, c_lib.divide_nda(self, other, c_int_array(other.shape)))
		elif type(other) is c_double:
			return DenseSparseArray(self.shape, c_lib.divide_f(self, other))
		else:
			return NotImplemented

	def convert_arg_for_c(self, other: DenseSparseArray | np.ndarray | float) -> DenseSparseArray | np.ndarray | c_double:
		""" convert the given argument to a form in which it is compatible with c, and
		    also compatible with self._as_attribute_ in terms of ndim and shape and all
		    that, so that they can reasonbaly operate with each other in the shadow realm
		"""
		if type(other) is DenseSparseArray:
			if self.dense_shape == other.dense_shape and self.sparse_shape == other.sparse_shape:
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
			return NotImplemented

	def __matmul__(self, other: DenseSparseArray | np.ndarray) -> DenseSparseArray | np.ndarray:
		if self.sparse_ndim != 1:
			raise ValueError("I've only implemented matrix multiplication for matrices with exactly 1 sparse dim")
		if self.shape[-1] != other.shape[0]:
			raise ValueError(f"the given shapes ({self.shape} and {other.shape}) aren't matrix-multiplication compatible")
		if type(other) is DenseSparseArray:
			if other.dense_ndim < 1:
				raise ValueError("can't matrix-multiply by a matrix with only sparse dims")
			return DenseSparseArray(self.shape[:-1] + other.shape[1:], c_lib.matmul_saa(self, other))
		elif type(other) is np.ndarray:
			return ndarray_from_c(
				c_lib.matmul_nda(self, other, c_int_array(other.shape), c_int(other.ndim)),
				self.shape[:-1] + other.shape[1:])
		else:
			return NotImplemented

	def __pow__(self, power: float) -> DenseSparseArray:
		return DenseSparseArray(self.shape, c_lib.power_f(self, c_double(power)))

	def __getitem__(self, index: tuple) -> DenseSparseArray:
		if type(index) is not tuple:
			raise NotImplementedError(f"this index is too complicated, {index!r}")
		# start by expanding any ...
		for i in range(len(index)):
			if index[i] is Ellipsis:
				index = index[:i] + (slice(None),)*(self.ndim - len(index) + 1) + index[i + 1:]
		if len(index) != self.ndim:
			raise IndexError(f"this index has {len(index)} indices but we can only index {self.ndim}")

		# then go thru and do each item one at a time
		shape = self.shape
		result = self
		for k in range(self.ndim - 1, -1, -1):
			if type(index[k]) is slice and index[k] == slice(None):
				pass
			elif k >= self.dense_ndim:
				raise NotImplementedError(f"this SparseArray implementation does not support indexing on the sparse axis: {index}")
			elif type(index[k]) == np.ndarray:
				if index[k].ndim == 1:
					if np.issubdtype(index[k].dtype, np.bool):
						indices = np.arange(self.shape[k])[index[k]]
					elif np.issubdtype(index[k].dtype, np.integer):
						indices = index[k]
					else:
						raise TypeError(f"don't kno how to interpret an index with an array of dtype {index[k].dtype}")
					shape = shape[:k] + (indices.size,) + shape[k+1:]
					result = DenseSparseArray(shape, c_lib.get_reindex_saa(
						result, c_int_array(indices), c_int(indices.size), c_int(k)))
				else:
					raise NotImplementedError("only 1D numpy ndarrays may be used to index")
			elif type(index[k]) == int:
				shape = shape[:k] + shape[k+1:]
				result = DenseSparseArray(shape, c_lib.get_slice_saa(result, c_int(index[k]), c_int(k)))
			else:
				raise NotImplementedError(f"I can't do this index, {index[k]!r}")
		return result

	def __str__(self) -> str:
		indented = str(np.array(self)).replace('\n', '\n ')
		return f"{self.dense_shape}x{self.sparse_shape}:\n {indented}"

	def __array__(self) -> np.ndarray:
		return ndarray_from_c(c_lib.to_dense(self, c_int_array(self.sparse_shape)), self.shape)

	def expand_dims(self, ndim: int):
		""" add several new dimensions to the front of this's shape.  all new dimensions
		    will have shape zero
		"""
		if ndim == 0:
			return self
		elif ndim > 0:
			return DenseSparseArray((1,)*ndim + self.shape, c_lib.expand_dims(self, ndim))

	def sum(self, axis: Collection[int] | int | None) -> DenseSparseArray | np.ndarray:
		if type(axis) is int:
			axis = [axis]
		if np.any(np.less(axis, 0)) or np.any(np.greater_equal(axis, self.ndim)):
			raise ValueError("all of the axes must be nonnegative (sorry)")
		_, instances = np.unique(axis, return_counts=True)
		if np.any(np.greater(instances, 1)):
			raise ValueError("the axes can't have duplicates")
		# if summing over all sparse axes, there's no need for the sparse data structure anymore
		if len(axis) == self.sparse_ndim and np.all(np.greater_equal(axis, self.dense_ndim)):
			return ndarray_from_c(c_lib.sum_all_sparse(self), self.dense_shape)
		elif np.any(np.greater_equal(axis, self.dense_ndim)):
			raise ValueError("I haven't implemented summing on individual sparse axes")
		# if summing over all dense axes, convert to a regular dense array
		if len(axis) == self.dense_ndim:
			return ndarray_from_c(c_lib.sum_all_dense(self, c_int_array(self.sparse_shape)), self.sparse_shape)
		# otherwise, do them one at a time
		else:
			result = self
			shape = list(self.shape[:])
			for k in sorted(axis, reverse=True):
				shape.pop(k)
				result = DenseSparseArray(shape, c_lib.sum_along_axis(result, c_int(k)))
			return result


if __name__ == "__main__":
	array = DenseSparseArray.from_coordinates(
		[10],
		np.array([[[[0], [1]],
		           [[0], [4]],
		           [[0], [5]]]]),
		np.array([[[1., -1.],
		           [0., 1.],
		           [3., -2.]]]))
	brray = DenseSparseArray.from_coordinates(
		[10],
		np.array([[[[0], [1]],
		           [[0], [4]],
		           [[5], [6]]]]),
		np.array([[[0., 1.],
		           [-2., 3.],
		           [-2., -2.]]])
	)
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
	print(array.expand_dims(2))
	print(array[0, :, :])
	print(array[:, 1, :])
	print(array[:, np.array([2, 1, 0]), :])
	print(array[0, np.array([2, 1, 0]), :])
	print(array.sum(axis=[0]))
	print(array.sum(axis=1))
	print(array.sum(axis=[0, 1]))
	print(array.sum(axis=0).sum(axis=[0]))
	print(array.sum(axis=2))

	amatrix = DenseSparseArray.from_coordinates(
		[3],
		np.array([[[2]], [[0]], [[1]], [[0]]]),
		np.array([[-1.], [3.], [2.], [1.]]),
	)
	bmatrix = brray[0, ...]
	cmatrix = np.array([[0., 1], [1, 0], [0, -1], [-3, 0], [0, 2], [-1, 0], [0, 3], [-2, 0], [0, -2], [2, 0]])
	# cmatrix = DenseSparseArray.from_coordinates(
	# 	[2, 2],
	# 	np.array([[[[0, 0], [1, 1]]], [[[0, 1], [1, 0]]], [[[0, 0], [1, 0]]], [[[0, 1], [1, 1]]], [[[0, 0], [0, 1]]], [[[1, 0], [1, 1]]], [[[0, 0], [1, 1]]], [[[0, 0], [1, 1]]], [[[0, 0], [1, 1]]], [[[0, 0], [1, 1]]]]),
	# 	np.array([[[-2.,     1.,]],   [[-1.,    -3.]],    [[ 2.,     1.]],    [[-2.,     1.,]],   [[-1.,    -3.]],    [[ 2.,     1.]],    [[-2.,     1.,]],   [[-1.,    -3.]],    [[ 2.,     1.]],     [[3.,    -1.]]]),
	# )

	print("A =", amatrix)
	print("B =", bmatrix)
	print("C =", cmatrix)
	print("AB = ", amatrix @ bmatrix)
	print("BC = ", bmatrix @ cmatrix)
	print("(AB)C = ", (amatrix @ bmatrix) @ cmatrix)
	print("A(BC) = ", amatrix @ (bmatrix @ cmatrix))

	eyes = DenseSparseArray.identity(([4, 2]))
	print(eyes)
