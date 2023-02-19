#!/usr/bin/env python
"""
sparse.py

a wrapper for the terrible scipy sparse array library that aims to fill in the rather substantial
gaps and make it actually usable
"""
from __future__ import annotations

import os
import sys
from ctypes import c_int, Structure, cdll, CDLL, POINTER, c_double
from functools import cache, cached_property
from typing import Sequence, cast, Collection, Optional

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.sparse import csr_array

from util import minimum_swaps

if sys.platform.startswith('win32'):
	c_lib = cdll.LoadLibrary("../lib/libsparse.dll")
elif sys.platform.startswith('linux'):
	c_lib = CDLL(os.path.join(os.getcwd(), "../lib/libsparse.so.1.0.0"))
else:
	raise OSError(f"I don't recognize the platform {sys.platform}")


# define a few useful types
c_double_p = POINTER(c_double)
c_int_p = POINTER(c_int)
class c_sparse(Structure):
	_fields_ = [("num_rows", c_int),
	            ("num_cols", c_int),
	            ("data", c_double_p),
	            ("indices", c_int_p),
	            ("indptr", c_int_p),
	            ]
def convert_to_struct(array: csr_array) -> c_sparse:
	""" convert a csr_array to a c struct """
	if not array.has_sorted_indices:
		raise ValueError("what does it take to get sorted indices around here?")
	return c_sparse(array.shape[0], array.shape[1],
	                array.data.ctypes.data_as(c_double_p),
	                array.indices.ctypes.data_as(c_int_p),
	                array.indptr.ctypes.data_as(c_int_p))

# declare all of the C functions we plan to use and their parameters' types
c_double_array = np.ctypeslib.ndpointer(dtype=c_double, ndim=1)
c_int_array = np.ctypeslib.ndpointer(dtype=c_int, ndim=1)
c_lib.reshape_matmul.argtypes = [c_sparse, c_sparse, c_double_array, c_int_array]
c_lib.reshape_matmul_indptr.argtypes = [c_sparse, c_sparse, c_int_array]
c_lib.elementwise_outer_product.argtypes = [c_sparse, c_sparse, c_double_array, c_int_array]
c_lib.repeat_diagonally.argtypes = [c_sparse, c_int, c_double_array, c_int_array]


class SparseNDArray:
	def __init__(self, csr: csr_array, shape: Sequence[int], sparse_ndim: int):
		""" an n-dimensional array whose underlying data is stored in the CSR format. """
		self.ndim = len(shape)
		self.shape = tuple(shape)
		self.size = np.product(shape, dtype=int)
		self.sparse_ndim = sparse_ndim
		self.sparse_shape = self.shape[-self.sparse_ndim:]
		self.sparse_size = np.product(self.sparse_shape, dtype=int)
		self.dense_ndim = self.ndim - self.sparse_ndim
		self.dense_shape = self.shape[:self.dense_ndim]
		self.dense_size = np.product(self.dense_shape, dtype=int)
		if type(csr) != csr_array:
			raise TypeError(f"this must not be a {type(csr)}")
		if csr.shape != (self.dense_size, self.sparse_size):
			raise ValueError(f"the given data does not match the requested shape")
		self.csr = csr

	@staticmethod
	def zeros(shape: Sequence[int], sparse_ndim: int) -> SparseNDArray:
		""" return an array of zeros with some dense axes and some empty sparse axes """
		return SparseNDArray(
			csr_array((
				np.product(shape[:-sparse_ndim], dtype=int),
				np.product(shape[-sparse_ndim:], dtype=int))),
			shape, sparse_ndim)

	@staticmethod
	def identity(shape: int | Sequence[int]) -> SparseNDArray:
		""" return an array where the dense part of the shape is the same as the sparse
		    part of the shape, and each sparse array contains a single 1 in the position
		    corresponding to its own position in the dense array.  this is the output
		    you would expect from identity(product(shape)).reshape((*shape, *shape)), but
		    I don't want to implement reshape.
		    :param shape: half the shape of the desired array (the shape of the space on which it operates)
		"""
		try:
			shape = tuple(shape)
		except TypeError:
			shape = (shape,)
		size = np.product(shape, dtype=int)
		return SparseNDArray(csr_array(sparse.identity(size, format="csr")), 2*shape, len(shape))

	@staticmethod
	def from_coordinates(sparse_shape: Sequence[int], indices: NDArray[int], values: NDArray[float]) -> SparseNDArray:
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
		dense_shape = values.shape[:-1]
		nnz_per_row = values.shape[-1]
		flat_indices = np.zeros(values.shape, dtype=int)
		for k in range(len(sparse_shape)):
			flat_indices = flat_indices*sparse_shape[k] + indices[..., k]
		indptr = np.arange(np.product(dense_shape, dtype=int) + 1)*nnz_per_row
		shape = np.concatenate([dense_shape, sparse_shape])
		csr = csr_array(
			(values.ravel(), flat_indices.ravel(), indptr),
			(np.product(indices.shape[:-2]), np.product(sparse_shape)))
		csr.eliminate_zeros()
		csr.sort_indices()
		csr.sum_duplicates()
		return SparseNDArray(csr, shape, len(sparse_shape))

	@staticmethod
	def from_dense(values: NDArray[float], sparse_ndim: int) -> SparseNDArray:
		""" return a DenseSparseArray that corresponds to this dense array and takes up as little
		    memory as possible.
		    :param values: the numpy array whose contents we want to copy
		    :param sparse_ndim: the number of sparse dimensions on the final array.  all dimensions
		                        of the result that are not sparse will be dense.
		"""
		if sparse_ndim > values.ndim:
			raise ValueError("the specified number of dense dimensions shouldn't be greater than the total number of dimensions")
		return SparseNDArray(csr_array(values.reshape((np.product(values.shape[:-sparse_ndim]), -1))),
		                     values.shape, sparse_ndim)

	@staticmethod
	def concatenate(elements: Sequence[SparseNDArray]) -> SparseNDArray:
		""" create a densesparsearray by stacking some existing ones verticly """
		corrected_elements = []
		for element in elements:
			if element.sparse_ndim != 1:
				raise NotImplementedError("I only concatenate arrays with 1 sparse dim")
			if element.ndim == 1 or element.ndim == 2:
				corrected_elements.append(element.csr)
			else:
				raise NotImplementedError("I only concatenate 2d arrays")
		csr = csr_array(sparse.vstack(corrected_elements, format="csr"))
		return SparseNDArray(csr, csr.shape, 1)

	def __add__(self, other: SparseNDArray) -> SparseNDArray:
		if type(other) is SparseNDArray:
			if self.dense_shape != other.dense_shape or self.shape != other.shape:
				raise ValueError(f"shapes do not match: {self.shape} + {other.shape}")
			return SparseNDArray(self.csr + other.csr, self.shape, self.sparse_ndim)
		else:
			return NotImplemented

	def __sub__(self, other: SparseNDArray) -> SparseNDArray:
		if type(other) is SparseNDArray:
			if self.dense_shape != other.dense_shape or self.sparse_shape != other.sparse_shape:
				raise ValueError(f"these array sizes do not match: {self.shape} != {other.shape}")
			return SparseNDArray(self.csr - other.csr, self.shape, self.sparse_ndim)
		else:
			return NotImplemented

	@cache
	def __neg__(self) -> SparseNDArray:
		return self * -1

	def __mul__(self, other: SparseNDArray | NDArray[float] | float) -> SparseNDArray:
		# multiplication by a scalar is simple and can bypass a bunch of this stuff.
		if np.ndim(other) == 0:
			return SparseNDArray(
				self.csr*other, self.shape, self.sparse_ndim)
		# multiplication by an NDArray may require some manual broadcasting
		elif type(other) is np.ndarray:
			if self.ndim != other.ndim:
				raise ValueError("arrays must have the same number of dimensions")
			other_dense_shape = other.shape[:self.dense_ndim]
			other_sparse_shape = other.shape[-self.sparse_ndim:]
			if np.product(other_dense_shape) == 1:  # there are some strict limits on what I’m willing to broadcast, tho
				broadcast_dense_shape = other_dense_shape
			else:
				broadcast_dense_shape = self.dense_shape
			if np.product(other_sparse_shape) == 1:
				broadcast_sparse_shape = other_sparse_shape
			else:
				raise ValueError("broadcasting like this would consume too much memory as I’ve implemented it.")
			if self.shape != other.shape:
				other = np.broadcast_to(other, broadcast_dense_shape + broadcast_sparse_shape)
			other = other.reshape(np.product(broadcast_dense_shape, dtype=int),
			                      np.product(broadcast_sparse_shape, dtype=int))
			return SparseNDArray( csr_array(self.csr*other), self.shape, self.sparse_ndim)
		# multiplication by a fellow sparse also supports a little bit of broadcasting... sort of
		elif type(other) is SparseNDArray:
			if self.sparse_shape != other.sparse_shape:
				raise ValueError(f"the sparse shapes don’t match: {self.sparse_shape} and {other.sparse_shape}")
			elif other.dense_size < self.dense_size:
				if other.csr.nnz == 0:
					return SparseNDArray.zeros(self.shape, self.sparse_ndim)
				else:
					raise ValueError("this broadcast is not implemented")
			elif other.dense_shape == self.dense_shape:
				return SparseNDArray(self.csr*other.csr, self.shape, self.sparse_ndim)
			elif other.dense_size > self.dense_size:
				return NotImplemented
			else:
				raise ValueError("I don’t know how to broadcast these dense shapes")
		else:
			return NotImplemented

	def __rmul__(self, other: SparseNDArray | NDArray[float] | float) -> SparseNDArray:
		return self * other

	def __truediv__(self, other: NDArray[float] | float) -> SparseNDArray:
		return self * (1/other)

	def __matmul__(self, other: SparseNDArray) -> SparseNDArray:
		if not hasattr(other, "shape"):
			return NotImplemented
		if self.shape[-1] != other.shape[0]:
			raise ValueError(f"the given shapes {self.dense_shape}+{self.sparse_shape} @ {other.shape} aren't matrix-multiplication compatible")
		if self.ndim > 2 or self.sparse_ndim != 1:
			raise ValueError("I haven’t defined matrix multiplication for nd-arrays in a way that makes sense.")
		out_shape = self.dense_shape + other.shape[1:]
		if type(other) is SparseNDArray:
			if other.shape[0]%self.shape[1] != 0:
				raise ValueError("the array dimensions cannot match.")
			c_a = convert_to_struct(self.csr)
			c_b = convert_to_struct(other.csr)
			row_sizes = np.empty(other.dense_size//self.sparse_size*self.dense_size, dtype=int)
			c_lib.reshape_matmul_indptr(c_a, c_b, row_sizes)
			indptrs = np.concatenate([[0], np.cumsum(row_sizes)])
			data = np.empty(indptrs[-1], dtype=float)
			indices = np.empty(indptrs[-1], dtype=int)
			c_lib.reshape_matmul(c_a, c_b, data, indices)
			result = csr_array((data, indices, indptrs), shape=(row_sizes.size, other.sparse_size))
			# result.sort_indices()
			# result.sum_duplicates()
			# result.eliminate_zeros()
			return SparseNDArray(result, out_shape, other.sparse_ndim)
		else:
			return NotImplemented

	def transpose(self) -> SparseNDArray:
		return SparseNDArray(csr_array(self.csr.transpose()), self.sparse_shape + self.dense_shape, self.dense_ndim)

	def expand_dims(self, ndim: int) -> SparseNDArray:
		return SparseNDArray(self.csr, (1,)*ndim + self.shape, self.sparse_ndim)

	def outer_multiply(self, other: SparseNDArray) -> SparseNDArray:
		""" multiply these where the dense axes are alined but the sparse axes are expanded """
		if self.dense_shape != other.dense_shape:
			raise ValueError(f"array shapes do not match: {self.dense_shape}, {other.dense_shape}")
		m = self.dense_size
		n = self.sparse_size*other.sparse_size
		a_nnz_per_row = np.diff(self.csr.indptr)
		b_nnz_per_row = np.diff(other.csr.indptr)
		row_pointers = np.concatenate([[0], np.cumsum(a_nnz_per_row*b_nnz_per_row)])
		values = np.empty(row_pointers[-1], dtype=float)
		indices = np.empty(row_pointers[-1], dtype=int)
		c_lib.elementwise_outer_product(convert_to_struct(self.csr), convert_to_struct(other.csr), values, indices)
		csr = csr_array((values, indices, row_pointers), shape=(m, n))
		return SparseNDArray(csr, self.dense_shape + self.sparse_shape + other.sparse_shape,
		                     self.sparse_ndim + other.sparse_ndim)

	def __pow__(self, power: float) -> SparseNDArray:
		return SparseNDArray(
			csr_array(
				(self.csr.data**power, self.csr.indices, self.csr.indptr),
				self.csr.shape),
			self.shape,
			self.sparse_ndim)

	@cache
	def __abs__(self) -> SparseNDArray:
		return SparseNDArray(abs(self.csr), self.shape, self.sparse_ndim)

	@cache
	def min(self) -> float:
		return self.csr.min()

	@cache
	def max(self) -> float:
		return self.csr.max()

	@cache
	def norm(self, orde: int) -> float:
		return np.linalg.norm(np.reshape(self.__array__(),
		                                 (self.dense_size, self.sparse_size)),
		                      ord=orde)  # just convert it to dense for this; I think it's fine

	@cache
	def transpose_matmul_self(self) -> SparseNDArray:
		""" this matrix's transpose times this matrix, cached """
		return self.transpose()@self

	def __getitem__(self, index: tuple) -> SparseNDArray:
		if type(index) is not tuple:
			index = (index,)
		for k in range(len(index)):
			if index[k] is Ellipsis:
				index = index[:k] + (slice(None),)*(self.ndim - len(index) + 1) + index[k + 1:]
		if len(index) != self.ndim:
			raise IndexError("you currently must index all the dimensions")
		new_shape = []
		rows = np.array([0.])
		for k, i in enumerate(index):
			if k < self.dense_ndim:
				rows = rows[:, np.newaxis]*self.shape[k]
				if np.issubdtype(type(i), np.integer):
					rows = rows + i
				elif type(i) is np.ndarray and np.issubdtype(cast(NDArray, i).dtype, np.integer):
					new_shape.append(len(i))
					rows = rows + i
				elif type(i) is np.ndarray and np.issubdtype(cast(NDArray, i).dtype, np.bool):
					new_shape.append(np.count_nonzero(i))
					rows = rows + np.nonzero(i)[0]
				elif type(i) is slice and i == slice(None):
					new_shape.append(self.shape[k])
					rows = rows + np.arange(self.shape[k])
				else:
					print(i)
					raise IndexError(f"I haven’t implemented such an index: {i}")
				rows = rows.ravel()
			else:
				if i != slice(None):
					raise ValueError("no indexing on the sparse axes!")
				new_shape.append(self.shape[k])
		return SparseNDArray(self.csr[rows, :], new_shape, self.sparse_ndim)

	@cached_property
	def T(self) -> SparseNDArray:
		return self.transpose()

	@cache
	def diagonal(self, k: int = 0) -> NDArray[float]:
		""" return a dense vector containing the items self[(i + k)%self.shape[0], i] for all i """
		return self.csr.diagonal(k)

	@cache
	def is_positive_definite(self) -> bool:
		""" determine whether or not this matrix is positive definite """
		if self.dense_shape != self.sparse_shape:
			raise ValueError("this is only for squares")
		lu = sparse.linalg.splu(self.csr.T)
		l_diag = lu.L.diagonal()
		u_diag = lu.U.diagonal()
		sign = np.sign(l_diag).prod()*np.sign(u_diag).prod()
		correction = (-1)**minimum_swaps(lu.perm_r)
		sign *= correction
		return sign > 0


	def inverse_matmul(self, other: NDArray[float]) -> NDArray[float]:
		""" compute self**-1 @ b for a symmetric positive definite matrix, iteratively.  this won't
			work if self is not symmetric, but it may work if it's positive definite
			:param other: the vector by which to multiply the inverse
			:return: the vector x that solves self@x = b
		"""
		if self.dense_ndim != 1 or self.sparse_ndim != 1 or self.dense_size != self.sparse_size:
			raise ValueError(f"this only works for square arrays")
		if self.dense_size != other.shape[0]:
			raise ValueError(f"the shapes don't match: {self.dense_shape}+{self.sparse_shape} × {other.shape}")
		return sparse.linalg.spsolve(self.csr, other)

	def __len__(self) -> int:
		if self.ndim > 0:
			return self.shape[0]
		else:
			raise ValueError("this array is not an array and has no len")

	def __str__(self) -> str:
		if self.sparse_size < 100:
			return str(self.__array__())
		else:
			return str(self.csr)

	def __array__(self) -> NDArray[float]:
		return self.csr.todense().reshape(self.shape)

	def to_array_array(self) -> NDArray[float]:
		""" convert all of the dense axes to numpy axes. the result is a numpy object array with dtype=DenseSparseArray
		    where each of the elements has no dense axes. this is useful for using more numpy syntax that I don't want
		    to program, but will be much slower than either a pure DenseSparseArray or a pure numpy ndarray.
		"""
		array = np.empty(self.dense_shape, dtype=SparseNDArray)
		for i in range(self.dense_size):
			index = np.unravel_index(i, self.dense_shape)
			array[index] = SparseNDArray(self.csr[i:i + 1, :], self.sparse_shape, self.sparse_ndim)
		return array

	def sum(self, axis: Optional[Collection[int]] = None) -> SparseNDArray | NDArray[float]:
		if type(axis) is int:
			axis = [axis]
		if axis is None or np.array_equal(axis, np.arange(self.ndim)):
			return self.csr.sum()
		elif np.array_equal(axis, np.arange(self.dense_ndim)):
			summation = csr_array(
				(np.ones(self.dense_size), np.arange(self.dense_size), [0, self.dense_size]),
				shape=(1, self.dense_size))
			return SparseNDArray(summation@self.csr, self.sparse_shape, self.sparse_ndim)
		elif np.array_equal(axis, np.arange(self.dense_ndim, self.ndim)):
			return self.csr.sum(axis=1).reshape(self.dense_shape)
		else:
			raise ValueError(f"I haven’t implemented summing a {self.dense_shape}×{self.sparse_shape} on axes {axis}")

	def reshape(self, shape, sparse_ndim) -> SparseNDArray:
		if np.product(shape, dtype=int) != self.size:
			raise ValueError(f"the requested new shape {shape} is not compatible with the old shape {self.shape}")
		dense_size = np.product(shape[:sparse_ndim], dtype=int)
		sparse_size = np.product(shape[sparse_ndim:], dtype=int)
		return SparseNDArray(csr_array(self.csr.reshape((dense_size, sparse_size))),
		                     shape, sparse_ndim)

	def repeat_diagonally(self, shape: Sequence[int]) -> SparseNDArray:
		""" okay, this method is pretty specific and kind of hard to explain... take a matrix and
		    append the given shape to the end of both the dense shape and the sparse shape.  the
		    result should be a linear operator that has the same effect, but applies to a flattened
		    vector instead of an nd one.
		"""
		size = np.product(shape, dtype=int)
		row_nnzs = np.repeat(np.diff(self.csr.indptr), size)
		indptr = np.concatenate([[0], np.cumsum(row_nnzs)])
		data = np.empty(indptr[-1], dtype=float)
		indices = np.empty(indptr[-1], dtype=int)
		c_lib.repeat_diagonally(convert_to_struct(self.csr), size, data, indices)
		return SparseNDArray(
			csr_array(
				(data, indices, indptr), (self.dense_size*size, self.sparse_size*size)),
			(self.dense_shape + tuple(shape) + self.sparse_shape + tuple(shape)),
			self.sparse_ndim + len(shape))


def test():
	array = SparseNDArray.from_coordinates(
		[10],
		np.array([[[[0], [1]],
		           [[0], [4]],
		           [[5], [6]]]]),
		np.array([[[1., -1.],
		           [0., 1.],
		           [3., -2.]]]))
	brray = SparseNDArray.from_coordinates(
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
	print("slice [0,...]:", array[0, :, :])
	print("slice [:,1,:]:", array[:, 1, :])
	print("reorder [2,1,0]:", array[:, np.array([2, 1, 0]), :])
	print("slice and reorder:", array[0, np.array([2, 1, 0]), :])
	print("sum dense axes:", array.sum(axis=[0, 1]))
	print("sum sparse axes:", array.sum(axis=[2]))
	print("sum all axes:", array.sum(axis=[0, 1, 2]))
	print("sum each axis:", array.sum(axis=[0, 1]).sum(axis=[0]))

	A = SparseNDArray.from_coordinates(
		[3],
		np.array([[[0], [1]], [[0], [2]], [[1], [2]]]),
		np.array([[ 1., -2.], [-2., -3.], [-3.,  4.]]),
	)
	b = np.array([-1., 4., 1.])
	print("A =", A)
	print("b =", b)
	print("Ab =", A@b)
	print("A^-1(Ab) =", A.inverse_matmul(A@b))
	print("A^T(b) =", A.T@b)
	print("diag(A) =", A.diagonal())

	amatrix = SparseNDArray.from_coordinates(
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
	print("A^TA = ", amatrix.T @ amatrix)

	eyes = SparseNDArray.identity([4, 2])
	print(eyes)

	print(SparseNDArray.identity(3))

	print(SparseNDArray.zeros((0, 6), 1))

if __name__ == "__main__":
	test()
