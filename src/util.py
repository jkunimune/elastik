#!/usr/bin/env python
"""
util.py

some handy utility functions that are used in multiple places
"""
from math import hypot, pi, cos, sin, inf, copysign, nan
from typing import Sequence, TYPE_CHECKING, Union, Iterable

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
	from sparse import SparseNDArray


class Scalar:
	def __init__(self, value):
		""" a float with a matmul method """
		self.value = value
		self.ndim = 0

	def __matmul__(self, other):
		return self.value * other

	def __rmatmul__(self, other):
		return self.value * other


class Ellipsoid:
	def __init__(self, a, f):
		""" a collection of values defining a spheroid """
		self.a = a
		""" major semi-axis """
		self.f = f
		""" flattening """
		self.b = a*(1 - f)
		""" minor semi-axis """
		self.R = a
		""" equatorial radius """
		self.e2 = 1 - (self.b/self.a)**2
		""" square of eccentricity """
		self.e = np.sqrt(self.e2)
		""" eccentricity """


Numeric = float | NDArray[float]
""" an object on which general arithmetic operators are defined """
Tensor = Union[NDArray[float], "SparseNDArray", Scalar]
""" an object that supports the matrix multiplication operator """
EARTH = Ellipsoid(6_378.137, 1/298.257_223_563)
""" the figure of the earth as given by WGS 84 """


def bin_centers(bin_edges: NDArray[float]) -> NDArray[float]:
	""" calculate the center of each bin """
	return (bin_edges[1:] + bin_edges[:-1])/2

def bin_index(x: float | NDArray[float], bin_edges: NDArray[float], right=False) -> int | NDArray[int]:
	""" I dislike the way numpy defines this function
	"""  # TODO: right side makes no sense
	return np.where(x < bin_edges[-1], np.digitize(x, bin_edges, right=right) - 1, bin_edges.size - 2)

def index_grid(shape: Sequence[int]) -> Sequence[NDArray[int]]:
	""" create a set of int matrices that together cover every index in an array of the given shape """
	indices = [np.arange(length) for length in shape]
	return np.meshgrid(*indices, indexing="ij")

def normalize(vector: NDArray[float]) -> NDArray[float]:
	""" normalize a vector such that it can be compared to other vectors on the basis of direction alone """
	if np.all(np.equal(vector, 0)):
		return vector
	else:
		return np.divide(vector, np.max(np.abs(vector)))

def vector_normalize(vector: NDArray[float]) -> NDArray[float]:
	""" normalize a vector such that its magnitude is one """
	if np.all(np.equal(vector, 0)):
		return vector
	else:
		return np.divide(vector, np.linalg.norm(vector))

def offset_from_angle(a: NDArray[float], b: NDArray[float], c: NDArray[float],
                      offset: float) -> NDArray[float]:
	""" find a point that is diagonally offset from an angle, in the direction it points """
	bend_direction = vector_normalize(c - b) - \
	                 vector_normalize(b - a)
	if hypot(*bend_direction) > 1e-4:
		bend_direction = vector_normalize(bend_direction)
	else:
		travel_direction = vector_normalize(c - b)
		bend_direction = np.array([-travel_direction[1], travel_direction[0]])
	return b - offset*bend_direction

def wrap_angle(x: Numeric, period=360) -> Numeric:
	""" wrap an angular value into the range [-period/2, period/2) """
	return x - np.floor((x + period/2)/period)*period

def to_cartesian(ф: Numeric, λ: Numeric) -> tuple[Numeric, Numeric, Numeric]:
	""" convert spherical coordinates in degrees to unit-sphere cartesian coordinates """
	return (np.cos(np.radians(ф))*np.cos(np.radians(λ)),
	        np.cos(np.radians(ф))*np.sin(np.radians(λ)),
	        np.sin(np.radians(ф)))

def rotation_matrix(angle: float) -> NDArray[float]:
	""" calculate a simple 2d rotation matrix """
	return np.array([[cos(angle), -sin(angle)],
	                 [sin(angle),  cos(angle)]])

def interp(x: Numeric, x0: float, x1: float, y0: Numeric, y1: Numeric):
	""" do linear interpolation given two points, and *not* assuming x1 >= x0 """
	return (x - x0)/(x1 - x0)*(y1 - y0) + y0

def dilate(x: NDArray[bool], distance: int) -> NDArray[bool]:
	""" take a 1D boolean array and make it so that any Falses near Trues become True.
	    then return the modified array.
	"""
	for i in range(distance):
		x[:-1] |= x[1:]
		x[1:] |= x[:-1]
	return x


def search_out_from(i0: int, j0: int, shape: tuple[int, int], max_distance: int) -> Iterable[tuple[int, int]]:
	""" yield a list of index pairs in the given shape orderd such that iterating thru the list spirals
	    outward from i0,j0.  it will be treated periodically on axis 1 (so j=0 is next to j=n-1) """
	option_grid = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
	option_list = np.reshape(np.stack(option_grid, axis=-1), (-1, 2))
	i_distance = abs(option_list[:, 0] - i0)
	j_distance = abs(option_list[:, 1] - j0)
	j_distance = np.minimum(j_distance, shape[1] - j_distance)  # account for periodicity
	distance = i_distance + j_distance
	order = np.argsort(distance)
	return option_list[order[distance[order] <= max_distance], :]


def intersects(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float], d: tuple[float, float]) -> bool:
	""" determine whether a--b intersects with c--d """
	denominator = ((a[0] - b[0])*(c[1] - d[1]) - (a[1] - b[1])*(c[0] - d[0]))
	if denominator == 0:
		return False
	for k in range(2):
		if a[k] == b[k]:
			intersection = a[k]
		elif c[k] == d[k]:
			intersection = c[k]
		else:
			intersection = ((a[0]*b[1] - a[1]*b[0])*(c[k] - d[k]) -
			                (a[k] - b[k])*(c[0]*d[1] - c[1]*d[0]))/denominator
		if not (min(a[k], b[k]) <= intersection <= max(a[k], b[k]) and
		        min(c[k], d[k]) <= intersection <= max(c[k], d[k])):
			return False
	return True

def fit_in_rectangle(polygon: NDArray[float]) -> tuple[float, tuple[float, float]]:
	""" find the smallest rectangle that contains the given polygon, and parameterize it with the
	    rotation and translation needed to make it landscape and centered on the origin.
	"""
	if polygon.ndim != 2 or polygon.shape[0] < 3 or polygon.shape[1] != 2:
		raise ValueError("the polygon must be a sequence of at least 3 points in 2-space")
	# start by finding the convex hull
	hull = convex_hull(polygon)
	best_transform = None
	best_area = inf
	# the best rectangle will be oriented along one of the convex hull edges
	for i in range(hull.shape[0]):
		# measure the angle
		if hull[i, 0] != hull[i - 1, 0]:
			angle = np.arctan(np.divide(*(hull[i, ::-1] - hull[i - 1, ::-1])))
		else:
			angle = pi/2
		rotated_hull = (rotation_matrix(-angle)@hull.T).T
		x_min, y_min = np.min(rotated_hull, axis=0)
		x_max, y_max = np.max(rotated_hull, axis=0)
		# measure the area
		area = (x_max - x_min)*(y_max - y_min)
		# take the one that has the smallest area
		if area < best_area:
			best_area = area
			x_center, y_center = (x_min + x_max)/2, (y_min + y_max)/2
			# (make it landscape)
			if x_max - x_min < y_max - y_min:
				x_center, y_center = copysign(y_center, angle), copysign(x_center, -angle)
				angle = angle - copysign(pi/2, angle)
			best_transform = -angle, (-x_center, -y_center)
	return best_transform

def rotate_and_shift(points: NDArray[float], rotation: float, shift: NDArray[float]) -> NDArray[float]:
	""" rotate some points about the origin and then translate them """
	if points.shape[-1] != 2:
		raise ValueError("the points must be in 2D space")
	points = np.moveaxis(points, -1, -2)  # for some reason the x/y axis must be -2 specificly for matmul
	rotated = rotation_matrix(rotation)@points
	rotated = np.moveaxis(rotated, -2, -1)
	return rotated + shift

def convex_hull(points: NDArray[float]) -> NDArray[float]:
	""" take a set of points and return a copy that is missing all of the points that are inside the
	    convex hull, and they're also widdershins-ordered now, using a graham scan.
	"""
	# first we must sort the points by angle
	x0, y0 = (np.min(points, axis=0) + np.max(points, axis=0))/2
	order = np.argsort(np.arctan2(points[:, 1] - y0, points[:, 0] - x0))
	points = points[order]
	# then cycle them around so they start on a point we know to be on the hull
	start = np.argmax(points[:, 0])
	points = np.concatenate([points[start:], points[:start]])
	# define a minor utility function
	def convex(a, b, c):
		return (c[0] - b[0])*(b[1] - a[1]) - (c[1] - b[1])*(b[0] - a[0]) < 0
	# go thru the polygon one thing at a time
	hull =  []
	for i in range(0, points.shape[0]):
		hull.append(points[i, :])
		# then, if the end is no longer convex, backtrace
		while len(hull) >= 3 and not convex(*hull[-3:]):
			hull.pop(-2)
	return np.array(hull)

def decimate_path(path: list[tuple[float, float]] | NDArray[float], resolution: float,
                  watch_for_longitude_wrapping=False) -> list[tuple[float, float]] | NDArray[float]:
	""" simplify a path in-place such that the number of nodes on each segment is only as
	    many as needed to make the curves look nice, using Ramer-Douglas
	"""
	if len(path) <= 2:
		return path
	path = np.array(path)

	# if this is on a globe, look for points where it’s jumping from one side to the other
	if watch_for_longitude_wrapping:
		wrapping_segments = abs(path[1:, 1] - path[0:-1, 1]) > 180
		# if you find one, do each hemisphere separately
		if np.any(wrapping_segments):
			wrap = np.nonzero(wrapping_segments)[0][0] + 1
			decimated_east = decimate_path(path[:wrap, :], resolution)
			decimated_west = decimate_path(path[wrap:, :], resolution, True)
			return np.concatenate([decimated_east, decimated_west])

	# otherwise, look for the point that is furthest from the strait-line representation of this path
	xA, yA = path[0, :]
	xB, yB = path[1:-1, :].T
	xC, yC = path[-1, :]
	ab = np.hypot(xB - xA, yB - yA)
	ac = np.hypot(xC - xA, yC - yA)
	ab_ac = (xB - xA)*(xC - xA) + (yB - yA)*(yC - yA)
	distance = np.sqrt(np.maximum(0, ab**2 - (ab_ac/ac)**2))
	# if it’s not so far, declare this safe to simplify
	if np.max(distance) < resolution:
		return [(xA, yA), (xC, yC)]
	# if it is too far, split the path there and call this function recursively on the two parts
	else:
		furthest = np.argmax(distance) + 1
		decimated_head = decimate_path(path[:furthest + 1, :], resolution)
		decimated_tail = decimate_path(path[furthest:, :], resolution)
		return np.concatenate([decimated_head[:-1], decimated_tail])


def simplify_path(path: Union[NDArray[float], "SparseNDArray"], cyclic=False
                  ) -> Union[NDArray[float], "SparseNDArray"]:
	""" simplify a path such that strait segments have no redundant midpoints marked in them, and
	    it does not retrace itself
	"""
	index = np.arange(len(path)) # instead of modifying the path directly, save some memory by just handling this index vector

	# start by looking for simple duplicates
	for i in range(index.size - 2, -1, -1):
		if np.array_equal(path[index[i], ...],
		                  path[index[i + 1], ...]):
			index = np.concatenate([index[:i], index[i + 1:]])
	if cyclic:
		while np.array_equal(path[index[0], ...],
		                     path[index[-1], ...]):
			index = index[:-1]

	# then check for retraced segments
	for i in range(index.size - 3, -1, -1):
		if i + 1 < index.size:
			if np.array_equal(path[index[i], ...],
			                  path[index[i + 2], ...]):
				index = np.concatenate([index[:i], index[i + 2:]])
	if cyclic:
		while np.array_equal(path[index[1], ...],
		                     path[index[-1], ...]):
			index = index[1:-1]

	# finally try to simplify over-resolved strait lines
	for i in range(index.size - 3, -1, -1):
		dr0 = normalize(np.subtract(path[index[i + 1], ...], path[index[i], ...]))
		dr1 = normalize(np.subtract(path[index[i + 2], ...], path[index[i + 1], ...]))
		if np.array_equal(dr0, dr1) or np.array_equal(dr0, -dr1):
			index = np.concatenate([index[:i + 1], index[i + 2:]])
	if cyclic:
		for i in [-1, -2]:
			dr0 = normalize(np.subtract(path[index[i + 1], ...], path[index[i], ...]))
			dr1 = normalize(np.subtract(path[index[i + 2], ...], path[index[i + 1], ...]))
			if np.array_equal(dr0, dr1) or np.array_equal(dr0, -dr1):
				index = index[1:] if i == -1 else index[:-1]

	return path[index, ...]


def refine_path(path: list[tuple[float, float]] | NDArray[float], resolution: float, period=np.inf) -> NDArray[float]:
	""" add points to a path such that it has no segments longer than resolution """
	i = 1
	while i < len(path):
		x0, y0 = path[i - 1]
		x1, y1 = path[i]
		if abs(y1 - y0) <= period/2:
			length = hypot(x1 - x0, y1 - y0)
			if length > resolution:
				path = np.concatenate([path[:i], [((min(x0, x1) + max(x0, x1))/2, (min(y0, y1) + max(y0, y1))/2)], path[i:]])
				i -= 1
		i += 1
	return path


def make_path_go_around_pole(path: list[tuple[float, float]] | NDArray[float]) -> NDArray[float]:
	""" add points to a path (in radians) on a globe such that it goes around the edges of
	    the domain rather than simply wrapping from -180° to 180° or vice versa.
	"""
	# first, identify the points where it crosses the antimeridian
	crossings = {}  # keys the index of the segment to +1 for the north pole and -1 for the south
	for i in range(1, path.shape[0]):
		if abs(path[i, 1] - path[i - 1, 1]) == 360:
			crossings[i] = 1 if path[i, 1] < path[i - 1, 1] else -1
	# if there are exactly two, set them up so they don't overlap
	if len(crossings) == 2:
		north_cross_index = max(crossings.keys(), key=lambda i: path[i, 0])
		south_cross_index = min(crossings.keys(), key=lambda i: path[i, 0])
		crossings[north_cross_index] = 1
		crossings[south_cross_index] = -1
	# then go replace each crossing
	for i, sign in reversed(sorted(crossings.items())):
		path = np.concatenate([path[:i],
		                       [[sign*90, path[i - 1, 1]]],
		                       [[sign*90, path[i, 1]]],
		                       path[i:]])
	return path


def inside_polygon(x: NDArray[float], y: NDArray[float], polygon: NDArray[float], convex=False) -> NDArray[bool]:
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


def inside_region(ф: Numeric, λ: Numeric, region: NDArray[float], period=360) -> NDArray[bool]:
	""" take a set of points on a globe and run a polygon containment test
	    :param ф: the latitude(s) in degrees, either for each point in question or for each
	              row if the relevant points are in a grid
	    :param λ: the longitude(s) in degrees, either for each point in question or for each
	              collum if the relevant points are in a grid
	    :param region: a polygon expressed as an n×2 array; each vertex is a row in degrees.
	                   if the polygon is not closed (i.e. the zeroth vertex equals the last
	                   one) its endpoints will get infinite vertical rays attached to them.
	    :param period: 360 for degrees and 2π for radians.
	    :return: a boolean array with the same shape as points (except the 2 at the end)
	             denoting whether each point is inside the region
	"""
	ф, λ = np.array(ф), np.array(λ)
	if ф.ndim == 1 and λ.ndim == 1 and ф.size != λ.size:
		ф, λ = ф[:, np.newaxis], λ[np.newaxis, :]
		out_shape = (ф.size, λ.size)
	else:
		out_shape = ф.shape
	# first we need to characterize the region so that we can classify points at untuched longitudes
	δλ_border = region[1:, 1] - region[:-1, 1]
	δλ_border[abs(δλ_border) > period/2] = nan
	ф_border = (region[1:, 0] + region[:-1, 0])/2
	inside_out = δλ_border[np.nanargmax(ф_border)] > 0 # for closed regions we have this nice trick

	# then we can bild up a precise bool mapping
	inside = np.full(out_shape, inside_out)
	nearest_segment = np.full(out_shape, np.inf)
	for i in reversed(range(1, region.shape[0])):  # the reversed happens to make it better in one particular edge case
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


def minimum_swaps(arr) -> int:
	""" Minimum number of swaps needed to order a permutation array """
	# from https://www.thepoorcoder.com/hackerrank-minimum-swaps-2-solution/
	a = dict(enumerate(arr))
	b = {v: k for k, v in a.items()}
	count = 0
	for i in a:
		x = a[i]
		if x != i:
			y = b[i]
			a[y] = x
			b[x] = y
			count += 1
	return count
