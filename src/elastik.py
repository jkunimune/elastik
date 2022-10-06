#!/usr/bin/env python
"""
elastik.py

this script is an example of how to use the Elastic projections. enclosed in this file is
everything you need to load the mesh files and project data onto them.
"""
from __future__ import annotations
from typing import Sequence, Iterable

import h5py
import numpy as np
import shapefile
from matplotlib import pyplot as plt

from sparse import DenseSparseArray


Line = list[tuple[float, float]]


def load_elastik_projection(name: str) -> list[Section]:
	""" load the hdf5 file that defines an elastik projection
	    :param name: one of "desa", "hai", or "lin"
	"""
	with h5py.File(f"../projection/elastik-{name}.h5", "r") as file:
		sections = []
		for h in range(file.attrs["num_sections"]):
			sections.append(Section(file[f"section{h}/latitude"][:],
			                        file[f"section{h}/longitude"][:],
			                        file[f"section{h}/projection"][:, :],
			                        file[f"section{h}/border"][:],
			                        ))
	return sections


def project(points: list[Line], projection: list[Section]) -> list[Line]:
	""" apply an Elastik projection, defined by a list of sections, to the given series of
	    latitudes and longitudes
	"""
	# first, set gradients at each of the nodes
	projected: list[Line] = []
	for i, line in enumerate(points):
		print(f"{i}/{len(points)}")
		for section in projection:
			sectioned_line = []
			for ф, λ in line:
				if section.contains(ф, λ): # TODO: cut it where it isn't contained and draw along the boundary
					sectioned_line.append((ф, λ))
			if len(sectioned_line) > 0:
				projected.append([])
				for ф, λ in sectioned_line: # TODO: vectorize once you've ensured that it works on points
					x, y = section.get_planar_coordinates(ф, λ)
					projected[-1].append((x, y))
	return projected



def inverse_project(points: np.ndarray, mesh: list[Section]) -> list[Line]:
	""" apply the inverse of an Elastik projection, defined by a list of sections, to find
	    the latitudes and longitudes that map to these locations on the map
	"""
	pass


def smooth_interpolate(xs: Sequence[float | np.ndarray], x_grids: Sequence[np.ndarray],
                       z_grid: np.ndarray | DenseSparseArray, dzdx_grids: Sequence[np.ndarray | DenseSparseArray],
                       differentiate=None) -> float | np.ndarray | DenseSparseArray:
	""" perform a bi-cubic interpolation of some quantity on a grid, using precomputed gradients
	    :param xs: the location vector that specifies where on the grid we want to look
	    :param x_grids: the x values at which the output is defined (each x_grid should be 1d)
	    :param z_grid: the known values at the grid intersection points
	    :param dzdx_grids: the known derivatives with respect to ф at the grid intersection points
	    :param differentiate: if set to a nonnegative number, this will return not the
	                          interpolated value at the given point, but the interpolated
	                          *derivative* with respect to one of the input coordinates.
	                          the value of the parameter indexes which coordinate along
	                          which to index
	    :return: the value z that fits at xs
	"""
	if len(xs) != len(x_grids) or len(x_grids) != len(dzdx_grids):
		raise ValueError("the number of dimensions is not consistent")
	ndim = len(xs)

	# choose a cell in the grid
	key = [np.interp(x, x_grid, np.arange(x_grid.size)) for x, x_grid in zip(xs, x_grids)]

	# calculate the cubic-interpolation weits for the adjacent values and gradients
	value_weits, slope_weits = [], []
	for k, i_full in enumerate(key):
		i = key[k] = np.minimum(np.floor(i_full).astype(int), x_grids[k].size - 2)
		ξ, ξ2, ξ3 = i_full - i, (i_full - i)**2, (i_full - i)**3
		dx = x_grids[k][i + 1] - x_grids[k][i]
		if differentiate is None or differentiate != k:
			value_weits.append([2*ξ3 - 3*ξ2 + 1, -2*ξ3 + 3*ξ2])
			slope_weits.append([(ξ3 - 2*ξ2 + ξ)*dx, (ξ3 - ξ2)*dx])
		else:
			value_weits.append([(6*ξ2 - 6*ξ)/dx, (-6*ξ2 + 6*ξ)/dx])
			slope_weits.append([3*ξ2 - 4*ξ + 1, 3*ξ2 - 2*ξ])
	value_weits = np.meshgrid(*value_weits, indexing="ij", sparse=True)
	slope_weits = np.meshgrid(*slope_weits, indexing="ij", sparse=True)

	# get the indexing all set up correctly
	index = tuple(np.meshgrid(*([i, i + 1] for i in key), indexing="ij", sparse=True))
	full = (slice(None),)*len(xs) + (np.newaxis,)*(z_grid.ndim - len(xs))

	# then multiply and combine all the things
	weits = product(value_weits)[full]
	result = np.sum(weits*z_grid[index], axis=tuple(range(ndim)))
	for k in range(ndim):
		weits = product(value_weits[:k] + [slope_weits[k]] + value_weits[k+1:])[full]
		result += np.sum(weits*dzdx_grids[k][index], axis=tuple(range(ndim)))
	return result


def gradient(Y: np.ndarray, x: np.ndarray, where: np.ndarary, axis: int) -> (np.ndarray[float], np.ndarray[bool]):
	""" Return the gradient of an N-dimensional array.

	    The gradient is computed using second ordre accurate central differences in the
	    interior points and eithre first or second ordre accurate one-sides (forward or
	    backwards) differences at the boundaries. The returned gradient hence has the
	    same shape as the input array.
	    :param Y: the values of which to take the gradient
	    :param x: the values by which we take the gradient in a 1d array whose length matches Y.shape[axis]
	    :param where: the values that are valid and should be consulted when calculating
	                  this gradient. it should have the same shape as Y. values of Y
	                  corresponding to a falsy in where will be ignored, and gradients
	                  centerd at such points will be markd as nan.
	    :param axis: the axis along which to take the gradient
	    :return: an array with the gradient values, as well as a boolean mask indicating
	             which of the returnd values are invalid (because there were not enuff valid inputs in the vicinity)
	"""
	if Y.shape[:where.ndim] != where.shape:
		raise ValueError("where must have the same shape as Y")
	# get ready for some numpy nonsense
	i = np.arange(x.size)
	# I roll the axes of Y such that the desired axis is axis 0
	new_axis_order = np.roll(np.arange(where.ndim), -axis)
	where = where.transpose(new_axis_order)
	Y = Y.transpose(np.concatenate([new_axis_order, np.arange(where.ndim, Y.ndim)]))
	# I add some nans to the end of axis 0 so that I don't haff to worry about index issues
	where = np.pad(where, [(0, 2), *[(0, 0)]*(where.ndim - 1)], constant_values=False)
	Y = np.pad(Y, [(0, 2), *[(0, 0)]*(Y.ndim - 1)], constant_values=np.nan)
	x = np.pad(x, [(0, 2)], constant_values=np.nan)
	# I set it up to broadcast properly
	while x.ndim < Y.ndim:
		x = np.expand_dims(x, axis=1)

	# then do this colossal nested where-statement
	centered_2nd = where[i, ...] &  where[i - 1, ...] &  where[i + 1, ...]
	backward_2nd = where[i, ...] &  where[i - 1, ...] & ~where[i + 1, ...] &  where[i - 2, ...]
	backward_1st = where[i, ...] &  where[i - 1, ...] & ~where[i + 1, ...] & ~where[i - 2, ...]
	forward_2nd =  where[i, ...] & ~where[i - 1, ...] &  where[i + 1, ...] &  where[i + 2, ...]
	forward_1st =  where[i, ...] & ~where[i - 1, ...] &  where[i + 1, ...] & ~where[i + 2, ...]
	impossible =  ~where[i, ...] | (~where[i - 1, ...] & ~where[i + 1, ...])
	methods = [(centered_2nd, (Y[i + 1, ...] - Y[i - 1, ...])/(x[i + 1] - x[i - 1])),
	           (backward_2nd, (3*Y[i, ...] - 4*Y[i - 1, ...] + Y[i - 2, ...])/(x[i] - x[i - 2])),
	           (backward_1st, (Y[i, ...] - Y[i - 1, ...])/(x[i] - x[i - 1])),
	           (forward_2nd, (3*Y[i, ...] - 4*Y[i + 1, ...] + Y[i + 2, ...])/(x[i] - x[i + 2])),
	           (forward_1st, (Y[i, ...] - Y[i + 1, ...])/(x[i] - x[i + 1])),
	           (impossible, np.nan*Y[i, ...])]
	grad = np.empty(Y[i, ...].shape, dtype=Y.dtype)
	for condition, formula in methods:
		grad[condition, ...] = formula[condition, ...]

	# finally, reset the axes
	old_axis_order = np.roll(np.arange(where.ndim), axis)
	return (grad.transpose(np.concatenate([old_axis_order, np.arange(where.ndim, Y.ndim)])),
            impossible.transpose(old_axis_order)) # I could use a maskedarray here, but it feels weerd to import the whole module just for this


def product(values: Iterable[np.ndarray | float | int]) -> np.ndarray | float | int:
	result = None
	for value in values:
		if result is None:
			result = value
		else:
			result = result*value
	return result


def load_geographic_data(filename: str) -> list[Line]:
	""" load a bunch of polylines from a shapefile
	    :param filename: valid values include "admin_0_countries", "coastline", "lakes",
	                     "land", "ocean", "rivers_lake_centerlines"
	"""
	lines: list[Line] = []
	with shapefile.Reader(f"../data/ne_110m_{filename}.zip") as shape_f:
		for shape in shape_f.shapes():
			lines.append([(ф, λ) for λ, ф in shape.points])
	return lines


class Section:
	def __init__(self, ф_nodes: np.ndarray, λ_nodes: np.ndarray,
	             xy_nodes: np.ndarray, border: np.ndarray):
		""" one lobe of an Elastik projection, containing a grid of latitudes and
		    longitudes as well as the corresponding x and y coordinates
		    :param ф_nodes: the node latitudes (deg)
		    :param λ_nodes: the node longitudes (deg)
		    :param xy_nodes: the grid of x- and y-values at each ф and λ (km)
		    :param border: the path that encloses the region this section defines (deg)
		"""
		self.ф_nodes = ф_nodes
		self.λ_nodes = λ_nodes
		self.xy_nodes = xy_nodes
		self.dxdф_nodes, _ = gradient(xy_nodes["x"], ф_nodes, where=np.isfinite(xy_nodes["x"]), axis=0)
		self.dxdλ_nodes, _ = gradient(xy_nodes["x"], λ_nodes, where=np.isfinite(xy_nodes["x"]), axis=1) # TODO: account for shared nodes, rite?
		self.dydф_nodes, _ = gradient(xy_nodes["y"], ф_nodes, where=np.isfinite(xy_nodes["y"]), axis=0)
		self.dydλ_nodes, _ = gradient(xy_nodes["y"], λ_nodes, where=np.isfinite(xy_nodes["y"]), axis=1)
		self.border = border


	def get_planar_coordinates(self, ф: np.ndarray | float, λ: np.ndarray | float) -> tuple[np.ndarray | float, np.ndarray | float]:
		""" take a point on the sphere and smoothly interpolate it to x and y """
		return (smooth_interpolate((ф, λ), (self.ф_nodes, self.λ_nodes),
		                           self.xy_nodes["x"], (self.dxdф_nodes, self.dxdλ_nodes)),
		        smooth_interpolate((ф, λ), (self.ф_nodes, self.λ_nodes),
		                           self.xy_nodes["y"], (self.dydф_nodes, self.dydλ_nodes)))


	def get_planar_gradients(self, ф: np.ndarray | float, λ: np.ndarray | float) -> np.ndarray:
		""" take a point on the sphere and calculate the 2×2 jacobian of its planar
		    coordinates with respect to its spherical ones
		"""
		return np.array([[
			smooth_interpolate((ф, λ), (self.ф_nodes, self.λ_nodes),
			                   self.xy_nodes["x"], (self.dxdф_nodes, self.dxdλ_nodes), differentiate=0),
			smooth_interpolate((ф, λ), (self.ф_nodes, self.λ_nodes),
			                   self.xy_nodes["x"], (self.dxdф_nodes, self.dxdλ_nodes), differentiate=1),
			smooth_interpolate((ф, λ), (self.ф_nodes, self.λ_nodes),
			                   self.xy_nodes["y"], (self.dydф_nodes, self.dydλ_nodes), differentiate=0),
			smooth_interpolate((ф, λ), (self.ф_nodes, self.λ_nodes),
			                   self.xy_nodes["y"], (self.dydф_nodes, self.dydλ_nodes), differentiate=1)]])

	def contains(self, ф: np.ndarray | float, λ: np.ndarray | float) -> np.ndarray | bool:
		nearest_segment = np.full(np.shape(ф), np.inf)
		inside = np.full(np.shape(ф), False)
		for i in range(1, self.border.shape[0]):
			ф0, λ0 = self.border[i - 1]
			ф1, λ1 = self.border[i]
			if λ1 != λ0 and abs(λ1 - λ0) <= 180:
				straddles = (λ0 <= λ) != (λ1 <= λ)
				фX = (λ - λ0)/(λ1 - λ0)*(ф1 - ф0) + ф0
				distance = np.where(straddles, abs(фX - ф), np.inf)
				inside = np.where(distance < nearest_segment, (λ1 > λ0) != (фX > ф), inside)
				nearest_segment = np.minimum(nearest_segment, distance)
		return inside


if __name__ == "__main__":
	data = load_geographic_data("coastline")
	sections = load_elastik_projection("hai")
	projected_data = project(data, sections)
	for i, line in enumerate(projected_data):
		plt.plot(*zip(*line), "k")
	plt.axis("equal")
	plt.axis("off")
	# fig.axen.get_xaxis().set_visible(False)
	# fig.axen.get_yaxis().set_visible(False)
	plt.savefig("../examples/tidal-phase.svg",
	            bbox_inches="tight", pad_inches=0)
	plt.show()

	# Z = np.random.random((5, 4))
	# Z[0, 0] = np.nan
	# Z[0, 3] = np.nan
	# Z[3, 1] = np.nan
	# x = np.linspace(0, 1, 5)
	# y = np.linspace(0, 1, 4)
	# dAdx = gradient(Z, x, axis=0)
	# dx = np.array([-.05, .05])
	# plt.figure()
	# for j in range(4):
	# 	plt.scatter(x, Z[:, j])
	# 	for i in range(5):
	# 		plt.plot(x[i] + dx, Z[i, j] + dAdx[i, j]*dx, f"C{j}")
	# dAdy = gradient(Z, y, axis=1)
	# dy = np.array([-.05, .05])
	# plt.figure()
	# for i in range(5):
	# 	plt.scatter(y, Z[i, :])
	# 	for j in range(4):
	# 		plt.plot(y[j] + dy, Z[i, j] + dAdy[i, j]*dy, f"C{i}")
	# plt.show()

