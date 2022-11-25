#!/usr/bin/env python
"""
sparse.py

a custom sparse array class.  it dips into the shadow realm to make the big calculations faster.
"""
from __future__ import annotations

import os
import sys
from ctypes import c_int, c_void_p, Structure, cdll, CDLL, Array, POINTER, c_double, c_bool, c_char
from typing import Callable, Sequence, Collection

import numpy as np
from numpy.typing import NDArray

if sys.platform.startswith('win32'):
	c_lib = cdll.LoadLibrary("../lib/libsparse.dll")
elif sys.platform.startswith('linux'):
	c_lib = CDLL(os.path.join(os.getcwd(), "../lib/libsparse.so.1.0.0"))
else:
	raise OSError(f"I don't recognize the platform {sys.platform}")


c_int_p = POINTER(c_int)
c_double_p = POINTER(c_double)
c_mut_char_p = POINTER(c_char) # does not automaticly copy to a Python bytes object, unlike c_char_p
c_ndarray = np.ctypeslib.ndpointer(dtype=c_double, flags='C_CONTIGUOUS')
c_int_ndarray = np.ctypeslib.ndpointer(dtype=c_int, flags='C_CONTIGUOUS')

class c_SparseArrayArray(Structure):
	_fields_ = [("ndim", c_int),
	            ("size", c_int),
	            ("shape", c_int_p),
	            ("elements", c_void_p),
	            ]

c_SparseArrayArray_p = POINTER(c_SparseArrayArray)

def c_int_array(lst: Sequence[int]) -> Array:
	return (c_int*len(lst))(*lst)

def c_SparseArrayArray_array(lst: Sequence[c_SparseArrayArray]) -> Array:
	return (c_SparseArrayArray*len(lst))(*lst)

def declare_c_func(func: Callable, args: list[type], res: type | None):
	func.argtypes = args
	func.restype = res

# declare all of the C functions we plan to use and their parameters' types
declare_c_func(c_lib.free_char_p, [c_mut_char_p], None)
declare_c_func(c_lib.free_saa, [c_SparseArrayArray], None)
declare_c_func(c_lib.free_nda, [c_ndarray], None)
declare_c_func(c_lib.add_saa, [c_SparseArrayArray, c_SparseArrayArray], c_SparseArrayArray)
declare_c_func(c_lib.subtract_saa, [c_SparseArrayArray, c_SparseArrayArray], c_SparseArrayArray)
declare_c_func(c_lib.multiply_saa, [c_SparseArrayArray, c_SparseArrayArray], c_SparseArrayArray)
declare_c_func(c_lib.matmul_saa, [c_SparseArrayArray, c_SparseArrayArray], c_SparseArrayArray)
declare_c_func(c_lib.outer_multiply_saa, [c_SparseArrayArray, c_SparseArrayArray], c_SparseArrayArray)
declare_c_func(c_lib.zeros, [c_int, c_int_p, c_int], c_SparseArrayArray)
declare_c_func(c_lib.identity, [c_int, c_int_p, c_bool], c_SparseArrayArray)
declare_c_func(c_lib.concatenate, [c_SparseArrayArray_p, c_int], c_SparseArrayArray)
declare_c_func(c_lib.new_saa, [c_int, c_int_p, c_int, c_int, c_int_ndarray, c_ndarray], c_SparseArrayArray)
declare_c_func(c_lib.multiply_nda, [c_SparseArrayArray, c_ndarray, c_int_p], c_SparseArrayArray)
declare_c_func(c_lib.divide_nda, [c_SparseArrayArray, c_ndarray, c_int_p], c_SparseArrayArray)
declare_c_func(c_lib.matmul_nda, [c_SparseArrayArray, c_ndarray, c_int_p, c_int, c_ndarray], None)
declare_c_func(c_lib.transpose_matmul_nda, [c_SparseArrayArray, c_int, c_ndarray, c_int_p, c_int, c_ndarray], None)
declare_c_func(c_lib.multiply_f, [c_SparseArrayArray, c_double], c_SparseArrayArray)
declare_c_func(c_lib.divide_f, [c_SparseArrayArray, c_double], c_SparseArrayArray)
declare_c_func(c_lib.power_f, [c_SparseArrayArray, c_double], c_SparseArrayArray)
declare_c_func(c_lib.sum_along_axis, [c_SparseArrayArray, c_int], c_SparseArrayArray)
declare_c_func(c_lib.sum_all_sparse, [c_SparseArrayArray, c_ndarray], None)
declare_c_func(c_lib.sum_all_dense, [c_SparseArrayArray, c_int_p, c_ndarray], None)
declare_c_func(c_lib.densify_axes, [c_SparseArrayArray, c_int_p, c_int], c_SparseArrayArray)
declare_c_func(c_lib.to_dense, [c_SparseArrayArray, c_int_p, c_int, c_ndarray], None)
declare_c_func(c_lib.expand_dims, [c_SparseArrayArray, c_int], c_SparseArrayArray)
declare_c_func(c_lib.get_slice_saa, [c_SparseArrayArray, c_int, c_int], c_SparseArrayArray)
declare_c_func(c_lib.get_diagonal_saa, [c_SparseArrayArray, c_ndarray], None)
declare_c_func(c_lib.get_reindex_saa, [c_SparseArrayArray, c_int_p, c_int, c_int], c_SparseArrayArray)
declare_c_func(c_lib.to_string, [c_SparseArrayArray], c_mut_char_p)


def str_from_c(data: c_mut_char_p | bytes) -> str:
	length = 0
	while data[length] != b"\0":
		length += 1
	string = bytes(data[:length]).decode("utf-8") # decode the existing memory
	c_lib.free_char_p(data) # then delete the old memory from c, since Python won't do it for us
	return string


class DenseSparseArray:
	def __init__(self, shape: Sequence[int], c_struct: c_SparseArrayArray):
		""" an array that is dense on the first few axes and sparse on the others. """
		if tuple(shape[:c_struct.ndim]) != tuple(c_struct.shape[i] for i in range(c_struct.ndim)):
			raise ValueError(f"the specified shape does not match the c object: {shape} vs {tuple(c_struct.shape[i] for i in range(c_struct.ndim))}")
		self.ndim = len(shape)
		self.shape = tuple(shape)
		self.size = np.product(shape)
		self.dense_ndim = c_struct.ndim
		self.dense_shape = self.shape[:self.dense_ndim]
		self.dense_size = np.product(self.dense_shape)
		self.sparse_ndim = self.ndim - self.dense_ndim
		self.sparse_shape = self.shape[self.dense_ndim:]
		self.sparse_size = np.product(self.sparse_shape)
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
	def identity(shape: int | Sequence[int], add_zero=False) -> DenseSparseArray:
		""" return an array where the dense part of the shape is the same as the sparse
		    part of the shape, and each sparse array contains a single 1 in the position
		    corresponding to its own position in the dense array.  this is the output
		    you would expect from identity(product(shape)).reshape((*shape, *shape)), but
		    I don't want to implement reshape.
		    :param shape: half the shape of the desired array (the shape of the space on which it operates)
		    :param add_zero: if set to True, an extra row will be added that's just zeros
		                     hey, sometimes that's useful!
		"""
		try:
			shape = tuple(shape)
		except TypeError:
			shape = (shape,)
		if add_zero:
			if len(shape) != 1:
				raise ValueError("add_zero can only be set if the identity matrix is 2d")
			return DenseSparseArray((shape[0] + 1, shape[0]),
			                        c_lib.identity(c_int(1), c_int_array(shape), c_bool(True)))
		else:
			return DenseSparseArray(shape*2,
			                        c_lib.identity(c_int(len(shape)), c_int_array(shape), c_bool(False)))

	@staticmethod
	def from_coordinates(sparse_shape: Sequence[int], indices: NDArray[int], values: NDArray[float]) -> DenseSparseArray:
		""" return an array where each SparseArray has the same number of nonzero values,
		    and they are located at explicitly known indices
		    :param sparse_shape: the shapes of the SparseArrays this contains
		    :param indices: the indices of the nonzero elements.  this should have shape
		                    (...n)?×m×k, where m is the number of elements in each
		                    SparseArray and k is the number of sparse dimensions. the
		                    shape up to that point corresponds to the dense shape.  these
		                    indices must be sorted and contain no duplicates if the
		                    DenseSparseArray is to operate elementwise on other
		                    DenseSparseArrays.  if you only want to operate on NDArray[float]s
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

	@staticmethod
	def concatenate(elements: Sequence[DenseSparseArray]) -> DenseSparseArray:
		""" create a densesparsearray by stacking some existing ones verticly """
		corrected_elements = []
		for element in elements:
			if element.sparse_ndim != 1:
				raise NotImplementedError("I only concatenate arrays with 1 sparse dim")
			if element.ndim == 1:
				corrected_elements.append(element.expand_dims(1))
			elif element.ndim == 2:
				corrected_elements.append(element)
			else:
				raise NotImplementedError("I only concatenate 2d arrays")
		shape = [0, corrected_elements[0].shape[1]]
		for element in corrected_elements:
			shape[0] += element.shape[0]
			if element.shape[1] != shape[1]:
				raise ValueError("you can't stack matrices unless they have matching shapes.")
		return DenseSparseArray(shape, c_lib.concatenate(
			c_SparseArrayArray_array([element._as_parameter_ for element in corrected_elements]),
			c_int(len(corrected_elements))))

	def __add__(self, other: DenseSparseArray | NDArray[float] | float) -> DenseSparseArray:
		if type(other) is DenseSparseArray:
			if self.dense_shape == other.dense_shape and self.sparse_shape == other.sparse_shape:
				return DenseSparseArray(self.shape, c_lib.add_saa(self, other))
			elif self.ndim == 0 and np.array(self) == 0:
				return other
			elif other.ndim == 0 and np.array(other) == 0:
				return self
			else:
				raise ValueError(f"these array sizes do not match (and neither is a scalar 0): {self.shape} != {other.shape}")
		elif type(other) is np.ndarray:
			if np.all(np.equal(other, 0)):
				return self
			else:
				raise TypeError("DenseSparseArrays cannot be added to normal arrays (unless the normal array is just 0)")
		else:
			return NotImplemented

	def __sub__(self, other: DenseSparseArray | NDArray[float] | float) -> DenseSparseArray:
		if type(other) is DenseSparseArray:
			if self.dense_shape == other.dense_shape and self.sparse_shape == other.sparse_shape:
				return DenseSparseArray(self.shape, c_lib.subtract_saa(self, other))
			elif other.ndim == 0 and np.array(other) == 0:
				return self
			elif self.ndim == 0 and np.array(self) == 0:
				return -other
			else:
				raise ValueError(f"these array sizes do not match (and neither is a scalar zero): {self.shape} != {other.shape}")
		elif type(other) is np.ndarray:
			if np.all(np.equal(other, 0)):
				return self
			else:
				raise TypeError("DenseSparseArrays cannot be subtracted from normal arrays (unless the normal array is just 0)")
		else:
			return NotImplemented

	def __neg__(self) -> DenseSparseArray:
		return self * -1

	def __mul__(self, other: DenseSparseArray | NDArray[float] | float) -> DenseSparseArray:
		other = self.convert_arg_for_c(other)
		if type(other) is DenseSparseArray:
			return DenseSparseArray(self.shape, c_lib.multiply_saa(self, other))
		if type(other) is np.ndarray:
			return DenseSparseArray(self.shape, c_lib.multiply_nda(self, other, c_int_array(other.shape)))
		elif type(other) is c_double:
			return DenseSparseArray(self.shape, c_lib.multiply_f(self, other))
		else:
			return NotImplemented

	def __rmul__(self, other: DenseSparseArray | NDArray[float] | float) -> DenseSparseArray:
		return self * other

	def __truediv__(self, other: NDArray[float] | float) -> DenseSparseArray:
		other = self.convert_arg_for_c(other)
		if type(other) is np.ndarray:
			return DenseSparseArray(self.shape, c_lib.divide_nda(self, other, c_int_array(other.shape)))
		elif type(other) is c_double:
			return DenseSparseArray(self.shape, c_lib.divide_f(self, other))
		else:
			return NotImplemented

	def convert_arg_for_c(self, other: DenseSparseArray | NDArray[float] | float) -> DenseSparseArray | NDArray[float] | c_double:
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
						if not other.flags["C_CONTIGUOUS"]:
							other = np.ascontiguousarray(other)
						sparse_axes = tuple(np.arange(self.dense_ndim, self.ndim))
						return np.squeeze(other, axis=sparse_axes).astype(float)
					else:
						raise NotImplementedError("I don't support elementwise operations on the sparse axes")
				else:
					raise IndexError(f"array shapes do not match: {self.shape} and {other.shape}")
			elif other.ndim == 0:
				return c_double(other[()])
			elif self.ndim == 0:
				return NotImplemented
			else:
				raise IndexError(f"array shapes do not match: {self.shape} and {other.shape}, {other.ndim}")
		elif np.issubdtype(type(other), np.number):
			return c_double(other)
		else:
			return NotImplemented

	def __matmul__(self, other: DenseSparseArray | NDArray[float]) -> DenseSparseArray | NDArray[float]:
		if not hasattr(other, "shape"):
			return NotImplemented
		if self.sparse_shape != other.shape[:self.sparse_ndim]:
			raise ValueError(f"the given shapes {self.dense_shape}+{self.sparse_shape} @ {other.shape} aren't matrix-multiplication compatible")
		if type(other) is DenseSparseArray:
			if other.dense_ndim < self.sparse_ndim:
				raise ValueError(f"the twoth array doesn't have enough dense dimensions; there must be at least {self.sparse_ndim}")
			return DenseSparseArray(self.shape[:-1] + other.shape[1:], c_lib.matmul_saa(self, other))
		elif type(other) is np.ndarray:
			if not other.flags["C_CONTIGUOUS"]:
				other = np.ascontiguousarray(other)
			result = np.empty(self.dense_shape + other.shape[self.sparse_ndim:])
			c_lib.matmul_nda(self, other, c_int_array(other.shape), c_int(other.ndim), result)
			return result
		else:
			return NotImplemented

	def transpose_matmul(self, other: NDArray[float]) -> NDArray[float]:
		""" the transpose of this array matrix multiplied by something """
		if self.dense_ndim != 1 or self.sparse_ndim != 1:
			raise ValueError("this is only designed to work for 2d matmulable matrices")
		if self.dense_size != other.shape[0]:
			raise ValueError(f"the given shapes {self.sparse_shape}+{self.dense_shape} @ {other.shape} aren't matrix-multiplication compatible")
		result = np.empty(self.sparse_shape + other.shape[1:])
		c_lib.transpose_matmul_nda(self, self.sparse_size, other, c_int_array(other.shape), c_int(other.ndim), result)
		return result

	def outer_multiply(self, other: DenseSparseArray) -> DenseSparseArray:
		if self.ndim == 0 and np.array(self) == 0:
			return self
		elif other.ndim == 0 and other.ndim == 0:
			return other
		elif self.dense_shape != other.dense_shape:
			raise ValueError(f"array shapes do not match: {self.dense_shape}, {other.dense_shape}")
		return DenseSparseArray(self.shape + other.sparse_shape, c_lib.outer_multiply_saa(self, other))

	def __pow__(self, power: float) -> DenseSparseArray:
		return DenseSparseArray(self.shape, c_lib.power_f(self, c_double(power)))

	def __getitem__(self, index: tuple) -> DenseSparseArray:
		if type(index) is not tuple:
			index = (index,)

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
			# slices do noting
			if type(index[k]) is slice and index[k] == slice(None):
				pass
			# non-slices on sparse dims thro an error
			elif k >= self.dense_ndim:
				raise NotImplementedError(f"this SparseArray implementation does not support indexing on the sparse axis: {index}")
			# int arrays rearrange an axis
			elif type(index[k]) == np.ndarray:
				if index[k].ndim == 1:
					if np.issubdtype(index[k].dtype, bool):
						if index[k].size != shape[k]:
							raise ValueError(f"the boolean array of length {index[k].size} doesn't match the axis length")
						indices = np.nonzero(index[k])[0]
					elif np.issubdtype(index[k].dtype, np.integer):
						indices = index[k]
					else:
						raise TypeError(f"don't kno how to interpret an index with an array of dtype {index[k].dtype}")
					shape = shape[:k] + (indices.size,) + shape[k+1:]
					result = DenseSparseArray(shape, c_lib.get_reindex_saa(
						result, c_int_array(indices), c_int(indices.size), c_int(k)))
				else:
					raise NotImplementedError("only 1d numpy ndarrays may be used to index")
			# integers pull out particular slices/items
			elif np.issubdtype(type(index[k]), np.integer):
				item = np.core.multiarray.normalize_axis_index(index[k], shape[k])
				shape = shape[:k] + shape[k+1:]
				result = DenseSparseArray(shape, c_lib.get_slice_saa(result, c_int(item), c_int(k)))
			# anything else is illegal
			else:
				raise NotImplementedError(f"I can't do this index, {index[k]!r}")
		return result

	def diagonal(self):
		""" return a dense vector containing the items self[i, i] for all i """
		if self.dense_shape != self.sparse_shape:
			raise ValueError(f"only square arrays have diagonals, not {self.dense_shape}x{self.sparse_shape}.")
		result = np.empty(self.dense_shape)
		c_lib.get_diagonal_saa(self, result)
		return result

	def inv_matmul(self, other: NDArray[float], tolerance=1e-8, damping=0) -> NDArray[float]:
		""" compute self**-1 @ b, iteratively
		    :param other: the vector by which to multiply the inverse
		    :param tolerance: the relative tolerance; the residual magnitude will be this factor
		                      less than b's magnitude.
		    :param damping: this doesn't actually compute self**-1 @ b; it <i>actually</i> computes
		                    (self + damping*I)**-1 @ b.  damping is a value that will be added to
		                    all of the diagonal elements.  increasing it will decrease the magnitude
		                    of the result, and when it is very large, the result will approach
		                    other/damping.
		    :return: the vector x that solves self@x = b
		"""
		if type(other) is not np.ndarray:
			return NotImplemented
		if self.dense_shape != self.sparse_shape:
			raise ValueError(f"only square matrices can be inverted, not {self.dense_shape}x{self.sparse_shape}")
		if other.shape != self.sparse_shape:
			raise ValueError("b must be a vector that matches self")
		absolute_tolerance = np.sum(other**2)*tolerance
		# diag = sign*self.diagonal() # TODO: I'm curius if replacing this with ones would make this faster
		guess = np.zeros(other.shape)#other/np.maximum(diag, np.quantile(abs(diag[diag != 0]), 1/sqrt(diag.size)))
		residue_old = other - self@guess - damping*guess
		direction = residue_old
		num_iterations = 0
		while True:
			Ad = self@direction + damping*direction
			α = np.sum(residue_old**2)/np.sum(direction*Ad)
			guess += α*direction
			residue_new = residue_old - α*Ad
			if np.sum(residue_new**2) < absolute_tolerance:
				if np.sum((other - self@guess - damping*guess)**2) < absolute_tolerance:
					return guess
			β = np.sum(residue_new**2)/np.sum(residue_old**2)
			direction = residue_new + β*direction
			residue_old = residue_new
			num_iterations += 1
			if num_iterations > 10*self.shape[0]:
				raise RuntimeError("conjugate gradients did not converge; we may be in a saddle region.")

	def __len__(self) -> int:
		if self.ndim > 0:
			return self.shape[0]
		else:
			raise ValueError("this array is not an array and has no len")

	def __str__(self) -> str:
		if self.sparse_size < 100:
			values = str(np.array(self))
		else:
			res = c_lib.to_string(self)
			values = str_from_c(res)
		values = values.replace('\n', '\n  ')
		return f"{self.dense_shape}x{self.sparse_shape}:\n  {values}"

	def __array__(self) -> NDArray[float]:
		result = np.empty(self.shape)
		c_lib.to_dense(self, c_int_array(self.sparse_shape), c_int(self.sparse_ndim), result)
		return result

	def to_array_array(self) -> NDArray[float]:
		""" convert all of the dense axes to numpy axes. the result is a numpy object array with dtype=DenseSparseArray
		    where each of the elements has no dense axes. this is useful for using more numpy syntax that I don't want
		    to program, but will be much slower than either a pure DenseSparseArray or a pure numpy ndarray.
		"""
		array = np.empty(self.dense_shape, dtype=DenseSparseArray)
		for i in range(self.dense_size):
			index = np.unravel_index(i, self.dense_shape)
			array[index] = self[(*index, ...)]
		return array

	def expand_dims(self, ndim: int):
		""" add several new dense dimensions to the front of this's shape.  all new
		    dimensions will have shape 1.
		    :param ndim: the number of dimensions to add to the front.
		"""
		if ndim == 0:
			return self
		elif ndim > 0:
			return DenseSparseArray((1,)*ndim + self.shape, c_lib.expand_dims(self, c_int(ndim)))

	def make_axes_dense(self, num_axes: int):
		""" convert the first few sparse dimensions of this to dense ones.
		    :param num_axes: the number of sparse dimensions to convert.
		"""
		return DenseSparseArray(self.shape, c_lib.densify_axes(
			self, c_int_array(self.sparse_shape[:num_axes]), c_int(num_axes)))

	def sum(self, axis: Collection[int] | int | None, make_dense=False) -> DenseSparseArray | NDArray[float]:
		if np.issubdtype(type(axis), np.integer):
			axis = [axis]
		if np.any(np.less(axis, 0)) or np.any(np.greater_equal(axis, self.ndim)):
			raise ValueError("all of the axes must be nonnegative (sorry)")
		_, instances = np.unique(axis, return_counts=True)
		if np.any(np.greater(instances, 1)):
			raise ValueError("the axes can't have duplicates")
		# if summing over all sparse axes, there's no need for the sparse data structure anymore
		if len(axis) == self.sparse_ndim and np.all(np.greater_equal(axis, self.dense_ndim)):
			result = np.empty(self.dense_shape)
			c_lib.sum_all_sparse(self, result)
			return result
		elif np.any(np.greater_equal(axis, self.dense_ndim)):
			raise ValueError("I haven't implemented summing on individual sparse axes")
		# if summing over all dense axes, convert to a regular dense array
		if len(axis) == self.dense_ndim and make_dense:
			result = np.empty(self.sparse_shape)
			c_lib.sum_all_dense(self, c_int_array(self.sparse_shape), result)
			return result
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
		           [[5], [6]]]]),
		np.array([[[1., -1.],
		           [0., 1.],
		           [3., -2.]]]))
	brray = DenseSparseArray.from_coordinates(
		[10],
		np.array([[[[0], [1]],
		           [[0], [4]],
		           [[0], [5]]]]),
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
	print("AxB =", array.outer_multiply(brray))
	print("BxA =", brray.outer_multiply(array))
	print("A*2 =", array*2)
	print("A/2 =", array/2)
	print("A^2 =", array**2)
	print("expand:", array.expand_dims(2))
	print("densify:", array.make_axes_dense(1))
	print("slice [0,...]:", array[0, :, :])
	print("slice [:,1,:]:", array[:, 1, :])
	print("reorder [2,1,0]:", array[:, np.array([2, 1, 0]), :])
	print("slice and reorder:", array[0, np.array([2, 1, 0]), :])
	print("sum axis 0:", array.sum(axis=[0]))
	print("sum axis 1:", array.sum(axis=1))
	print("sum both:", array.sum(axis=[0, 1]))
	print("sum each:", array.sum(axis=0).sum(axis=[0]))
	print("sum axis 2:", array.sum(axis=2))

	A = DenseSparseArray.from_coordinates(
		[3],
		np.array([[[0], [1]], [[0], [1]], [[2], [0]]]),
		np.array([[ 1.,  0.], [-2.,  3.], [ 4., -5.]]),
	)
	b = np.array([-1., 4., 1.])
	print("A =", A)
	print("b =", b)
	print("Ab =", A@b)
	print("A^-1(Ab) =", A.inv_matmul(A@b))
	print("A^T(b) =", A.transpose_matmul(b))
	print("diag(A) =", A.diagonal())

	amatrix = DenseSparseArray.from_coordinates(
		[3],
		np.array([[[2], [0]], [[0], [0]], [[1], [0]], [[0], [0]]]),
		np.array([[-1., 0], [3., 0], [2., -2], [1., 0]]),
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

	eyes = DenseSparseArray.identity([4, 2])
	print(eyes)

	print(DenseSparseArray.identity(3, add_zero=True))

	print(DenseSparseArray.zeros((0,), (6,)))
