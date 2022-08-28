#!/usr/bin/env python
"""
util.py

some handy utility functions that are used in multiple places
"""
from math import hypot, pi
from typing import Sequence, Iterable

import h5py
import numpy as np

from sparse import DenseSparseArray


class Ellipsoid:
	def __init__(self, a, f):
		""" a collection of values defining a spheroid """
		self.a = a
		""" major semiaxis """
		self.f = f
		""" flattening """
		self.b = a*(1 - f)
		""" minor semiaxis """
		self.R = a
		""" equatorial radius """
		self.e2 = 1 - (self.b/self.a)**2
		""" square of eccentricity """
		self.e = np.sqrt(self.e2)
		""" eccentricity """

class Scalar:
	def __init__(self, value):
		""" a float with a matmul method """
		self.value = value

	def __matmul__(self, other):
		return self.value * other

	def __rmatmul__(self, other):
		return self.value * other


h5_str = h5py.string_dtype(encoding='utf-8')
""" a str type compatible with h5py """

h5_xy_tuple = [("x", float), ("y", float)]
""" an (x, y) type compatible with h5py """

h5_фλ_tuple = [("latitude", float), ("longitude", float)]
""" a (ф, λ) type compatible with h5py """

EARTH = Ellipsoid(6_378.137, 1/298.257_223_563)
""" the figure of the earth as given by WGS 84 """


def bin_centers(bin_edges: np.ndarray) -> np.ndarray:
	""" calculate the center of each bin """
	return (bin_edges[1:] + bin_edges[:-1])/2

def bin_index(x: float | np.ndarray, bin_edges: np.ndarray) -> float | np.ndarray:
	""" I dislike the way numpy defines this function
	    :param: coerce whether to force everything into the bins (useful when there's roundoff)
	"""
	return np.where(x < bin_edges[-1], np.digitize(x, bin_edges) - 1, bin_edges.size - 2)

def index_grid(shape: Sequence[int]) -> Sequence[np.ndarray]:
	""" create a set of int matrices that together cover every index in an array of the given shape """
	indices = [np.arange(length) for length in shape]
	return np.meshgrid(*indices, indexing="ij")

def normalize(vector: Sequence[float]) -> Sequence[float]:
	""" normalize a vector such that it can be compared to other vectors on the basis of direction alone """
	if np.all(np.equal(vector, 0)):
		return vector
	else:
		return np.divide(vector, abs(vector[np.argmax(np.abs(vector))]))

def wrap_angle(x: float | np.ndarray) -> float | np.ndarray: # TODO: come up with a better name
	""" wrap an angular value into the range [-np.pi, np.pi) """
	return x - np.floor((x + np.pi)/(2*np.pi))*2*np.pi

def to_cartesian(ф: float | np.ndarray, λ: float | np.ndarray) -> tuple[float | np.ndarray, float | np.ndarray, float | np.ndarray]:
	""" convert spherical coordinates in degrees to unit-sphere cartesian coordinates """
	return (np.cos(np.radians(ф))*np.cos(np.radians(λ)),
	        np.cos(np.radians(ф))*np.sin(np.radians(λ)),
	        np.sin(np.radians(ф)))

def interp(x: float, x0: float, x1: float, y0: float, y1: float):
	""" do linear interpolation given two points, and *not* assuming x1 >= x0 """
	return (x - x0)/(x1 - x0)*(y1 - y0) + y0

def dilate(x: np.ndarray, distance: int) -> np.ndarray:
	""" take a 1D boolean array and make it so that any Falses near Trues become True.
	    then return the modified array.
	"""
	for i in range(distance):
		x[:-1] |= x[1:]
		x[1:] |= x[:-1]
	return x

def simplify_path(path: list[Iterable[float]] | np.ndarray | DenseSparseArray) -> list[Iterable[float]] | np.ndarray | DenseSparseArray:
	""" simplify a path in-place such that strait segments have no redundant midpoints
	    marked in them, and it does not retrace itself
	"""
	def num_points():
		try:
			return path.shape[0]
		except TypeError:
			return len(path)

	for i in range(num_points() - 2, 0, -1):
		r0 = path[i - 1, ...]
		r1 = path[i, ...]
		r2 = path[i + 1, ...]

		# start by looking for retraced segments
		if np.array_equal(r0, r2):
			index = np.concatenate([np.arange(i), np.arange(i + 2, num_points())])
			path = path[index, ...]

		# then try to simplify the lines
		elif np.array_equal(normalize(np.subtract(r2, r1)), normalize(np.subtract(r1, r0))):
			index = np.concatenate([np.arange(i), np.arange(i + 1, num_points())])
			path = path[index, ...]

	return path


def refine_path(path: list[tuple[float, float]] | np.ndarray, resolution: float, period=np.inf) -> np.ndarray:
	""" add points to a path such that it has no segments longer than resolution """
	i = 0
	while i < len(path):
		x0, y0 = path[i - 1]
		x1, y1 = path[i]
		if abs(y1 - y0) <= period/2:
			length = hypot(x1 - x0, y1 - y0)
			if length > resolution:
				path = np.concatenate([path[:i], [((x0 + x1)/2, (y0 + y1)/2)], path[i:]])
				i -= 1
		i += 1
	return path


def inside_polygon(x: np.ndarray, y: np.ndarray, polygon: np.ndarray, convex=False):
	""" take a set of points in the plane and run a polygon containment test
	    :param x: the x coordinates of the points
	    :param y: the y coordinates of the points
	    :param polygon: the vertices of the polygon given as an n×2 array
	    :param convex: if set to true, this will use a simpler algorithm that assumes the polygon is convex
	"""
	if not convex:
		raise NotImplementedError("this isn't implemented because I'm lazy")
	inside = np.full(np.shape(x), True)
	for i in range(polygon.shape[0]):
		x0, y0 = polygon[i - 1, :]
		x1, y1 = polygon[i, :]
		inside &= (x - x0)*(y - y1) - (x - x1)*(y - y0) > 0
	return inside


def inside_region(ф: np.ndarray, λ: np.ndarray, region: np.ndarray, period=360) -> np.ndarray:
	""" take a set of points on a globe and run a polygon containment test
	    :param ф: the latitudes in degrees, either for each point in question or for each
	              row if the relevant points are in a grid
	    :param λ: the longitudes in degrees, either for each point in question or for each
	              collum if the relevant points are in a grid
	    :param region: a polygon expressed as an n×2 array; each vertex is a row in degrees.
	                   if the polygon is not closed (i.e. the zeroth vertex equals the last
	                   one) its endpoitns will get infinite vertical rays attachd to them.
	    :param period: 360 for degrees and 2π for radians.
	    :return: a boolean array with the same shape as points (except the 2 at the end)
	             denoting whether each point is inside the region
	"""
	if ф.ndim == 1 and λ.ndim == 1 and ф.size != λ.size:
		ф, λ = ф[:, np.newaxis], λ[np.newaxis, :]
		out_shape = (ф.size, λ.size)
	else:
		out_shape = ф.shape
	# first we need to haracterize the region so that we can classify points at untuchd longitudes
	δλ_border = region[1:, 1] - region[:-1, 1]
	ф_border = region[1:, 0]
	if np.array_equal(region[0, :], region[-1, :]):
		inside_out = δλ_border[np.argmax(ф_border[:-1] + ф_border[1:])] > 0 # for closed regions we have this nice trick
	else:
		endpoints_down = np.mean(ф_border) > (ф_border[0] + ф_border[-1])/2
		goes_east = np.sum(δλ_border, where=abs(δλ_border) <= period/2) > 0
		inside_out = endpoints_down == goes_east # for open regions we need this heuristic

	# then we can bild up a precise bool mapping
	inside = np.full(out_shape, inside_out)
	nearest_segment = np.full(out_shape, np.inf)
	for i in range(1, region.shape[0]):
		ф0, λ0 = region[i - 1, :]
		ф1, λ1 = region[i, :]
		# this is a nonzero model based on virtual vertical rays drawn from each queried point
		if λ1 != λ0 and abs(λ1 - λ0) <= period/2:
			straddles = (λ0 <= λ) != (λ1 <= λ)
			фX = interp(λ, λ0, λ1, ф0, ф1)
			Δф = abs(фX - ф)
			affected = straddles & (Δф < nearest_segment)
			inside[affected] = (λ1 > λ0) != (фX > ф)[affected]
			nearest_segment[affected] = Δф[affected]
	return inside
