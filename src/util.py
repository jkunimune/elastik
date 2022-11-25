#!/usr/bin/env python
"""
util.py

some handy utility functions that are used in multiple places
"""
from math import hypot, pi, cos, sin, inf, copysign, sqrt
from typing import Sequence, Iterable

import numpy as np
from numpy.typing import NDArray

from sparse import DenseSparseArray


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
Tensor = NDArray[float] | DenseSparseArray | Scalar
""" an object that supports the matrix multiplication operator """
EARTH = Ellipsoid(6_378.137, 1/298.257_223_563)
""" the figure of the earth as given by WGS 84 """


def bin_centers(bin_edges: NDArray[float]) -> NDArray[float]:
	""" calculate the center of each bin """
	return (bin_edges[1:] + bin_edges[:-1])/2

def bin_index(x: float | NDArray[float], bin_edges: NDArray[float]) -> float | NDArray[int]:
	""" I dislike the way numpy defines this function
	    :param: coerce whether to force everything into the bins (useful when there's round-off)
	"""
	return np.where(x < bin_edges[-1], np.digitize(x, bin_edges) - 1, bin_edges.size - 2)

def index_grid(shape: Sequence[int]) -> Sequence[np.ndarray]:
	""" create a set of int matrices that together cover every index in an array of the given shape """
	indices = [np.arange(length) for length in shape]
	return np.meshgrid(*indices, indexing="ij")

def normalize(vector: NDArray[float]) -> NDArray[float]:
	""" normalize a vector such that it can be compared to other vectors on the basis of direction alone """
	if np.all(np.equal(vector, 0)):
		return vector
	else:
		return np.divide(vector, abs(vector[np.argmax(np.abs(vector))]))

def wrap_angle(x: Numeric) -> Numeric:
	""" wrap an angular value into the range [-pi, pi) """
	return x - np.floor((x + pi)/(2*pi))*2*pi

def to_cartesian(ф: Numeric, λ: Numeric) -> tuple[Numeric, Numeric, Numeric]:
	""" convert spherical coordinates in degrees to unit-sphere cartesian coordinates """
	return (np.cos(np.radians(ф))*np.cos(np.radians(λ)),
	        np.cos(np.radians(ф))*np.sin(np.radians(λ)),
	        np.sin(np.radians(ф)))

def rotation_matrix(angle: float) -> NDArray[float]:
	""" calculate a simple 2d rotation matrix """
	return np.array([[cos(angle), -sin(angle)],
	                 [sin(angle),  cos(angle)]])

def interp(x: Numeric, x0: float, x1: float, y0: float, y1: float):
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

def fit_in_rectangle(polygon: NDArray[float]) -> tuple[float, tuple[float, float]]:
	""" find the smallest rectangle that contains the given polygon, and parameterize it with the
	    rotation and translation needed to make it landscape and centered on the origin.
	"""
	if polygon.ndim != 2 or polygon.shape[0] < 3 or polygon.shape[1] != 2:
		raise ValueError("the polygon must be a sequence of at least 3 point sin 2-space")
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
	if points.ndim != 2 or points.shape[1] != 2:
		raise ValueError("the polygon must be a sequence of at least 3 point sin 2-space")
	return (rotation_matrix(rotation)@points.T).T + shift

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

def decimate_path(path: list[tuple[float, float]] | NDArray[float], resolution: float) -> list[tuple[float, float]] | NDArray[float]:
	""" simplify a path in-place such that the number of nodes on each segment is only as
	    many as needed to make the curves look nice, using Ramer-Douglas
	"""
	if len(path) <= 2:
		return path
	path = np.array(path)
	xA, yA = path[0, :]
	xB, yB = path[1:-1, :].T
	xC, yC = path[-1, :]
	ab = np.hypot(xB - xA, yB - yA)
	ac = np.hypot(xC - xA, yC - yA)
	ab_ac = (xB - xA)*(xC - xA) + (yB - yA)*(yC - yA)
	distance = np.sqrt(np.maximum(0, ab**2 - (ab_ac/ac)**2))
	if np.max(distance) < resolution:
		return [(xA, yA), (xC, yC)]
	else:
		furthest = np.argmax(distance) + 1
		decimated_head = decimate_path(path[:furthest + 1, :], resolution)
		decimated_tail = decimate_path(path[furthest:, :], resolution)
		return np.concatenate([decimated_head, decimated_tail])


def simplify_path(path: list[Iterable[float]] | NDArray[float] | DenseSparseArray, cyclic=False) -> list[Iterable[float]] | NDArray[float] | DenseSparseArray:
	""" simplify a path in-place such that strait segments have no redundant midpoints
	    marked in them, and it does not retrace itself
	"""
	index = np.arange(len(path)) # instead of modifying the path directly, save some memory by just handling this index vector

	# start by looking for simple duplicates
	for i in range(index.size - 2, -1, -1):
		if np.array_equal(path[index[i], ...],
		                  path[index[i + 1], ...]):
			index = np.concatenate([index[:i], index[i + 1:]])
	while cyclic and np.array_equal(path[index[0], ...],
	                                path[index[-1], ...]):
		index = index[:-1]

	# then check for retraced segments
	for i in range(index.size - 3, -1, -1):
		if i + 1 < index.size:
			if np.array_equal(path[index[i], ...],
			                  path[index[i + 2], ...]):
				index = np.concatenate([index[:i], index[i + 2:]])
	while cyclic and np.array_equal(path[index[1], ...],
	                                path[index[-1], ...]):
		index = index[1:-1]

	# finally try to simplify over-resolved strait lines
	for i in range(index.size - 3, -1, -1):
		r0 = path[index[i], ...]
		r1 = path[index[i + 1], ...]
		r2 = path[index[i + 2], ...]
		if np.array_equal(normalize(np.subtract(r2, r1)), normalize(np.subtract(r1, r0))):
			index = np.concatenate([index[:i + 1], index[i + 2:]])

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


def inside_region(ф: NDArray[float], λ: NDArray[float], region: NDArray[float], period=360) -> NDArray[bool]:
	""" take a set of points on a globe and run a polygon containment test
	    :param ф: the latitudes in degrees, either for each point in question or for each
	              row if the relevant points are in a grid
	    :param λ: the longitudes in degrees, either for each point in question or for each
	              collum if the relevant points are in a grid
	    :param region: a polygon expressed as an n×2 array; each vertex is a row in degrees.
	                   if the polygon is not closed (i.e. the zeroth vertex equals the last
	                   one) its endpoints will get infinite vertical rays attached to them.
	    :param period: 360 for degrees and 2π for radians.
	    :return: a boolean array with the same shape as points (except the 2 at the end)
	             denoting whether each point is inside the region
	"""
	if ф.ndim == 1 and λ.ndim == 1 and ф.size != λ.size:
		ф, λ = ф[:, np.newaxis], λ[np.newaxis, :]
		out_shape = (ф.size, λ.size)
	else:
		out_shape = ф.shape
	# first we need to characterize the region so that we can classify points at untuched longitudes
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


def polytope_project(point: NDArray[float], polytope_mat: DenseSparseArray, polytope_lim: float | NDArray[float],
                     tolerance: float, certainty: float = 60) -> NDArray[float]:
	""" project a given point onto a polytope defined by the inequality
	        all(ploytope_mat@point <= polytope_lim + tolerance)
	    I learned this fast dual-based proximal gradient strategy from
	        Beck, A. & Teboulle, M. "A fast dual proximal gradient algorithm for
	        convex minimization and applications", <i>Operations Research Letters</i> <b>42</b> 1
	        (2014), p. 1–6. doi:10.1016/j.orl.2013.10.007,
	    :param point: the point to project
	    :param polytope_mat: the matrix that defines the normals of the polytope faces
	    :param polytope_lim: the quantity that defines the size of the polytope.  it may be an array
	                         if point is 2d.  a point is in the polytope iff
	                         polytope_mat @ point[:, k] <= polytope_lim[k] for all k
	    :param tolerance: how far outside of the polytope a returned point may be
	    :param certainty: how many iterations it should try once it's within the tolerance to ensure
	                      it's finding the best point
	"""
	if point.ndim == 2:
		if point.shape[1] != polytope_lim.size:
			raise ValueError(f"if you want this fancy functionality, the shapes must match")
		# if there are multiple dimensions, do each dimension one at a time
		return np.stack([polytope_project(point[:, k], polytope_mat, polytope_lim[k], tolerance, certainty)
		                 for k in range(point.shape[1])]).T
	elif point.ndim != 1 or np.ndim(polytope_lim) != 0:
		raise ValueError(f"I don't think this works with {point.ndim}d-arrays instead of (1d) vectors")
	if polytope_mat.ndim != 2:
		raise ValueError(f"the matrix should be a (2d) matrix, not a {polytope_mat.ndim}d-array")
	if point.shape[0] != polytope_mat.shape[1]:
		raise ValueError("these polytope definitions don't jive")
	# check to see if we're already done
	if np.all(polytope_mat@point <= polytope_lim):
		return point

	# try not to have the tolerance be bigger than the current position; it screws with stuff
	tolerance = min(tolerance, np.max(polytope_mat@point - polytope_lim)*1e-1)

	# establish the parameters and persisting variables
	L = np.linalg.norm(polytope_mat, ord=2)**2
	x_new = None
	w_old = y_old = np.zeros(polytope_mat.shape[0])
	t_old = 1
	candidates = []
	# loop thru the proximal gradient descent of the dual problem
	for i in range(1_000):
		grad_F = polytope_mat@(point + polytope_mat.transpose_matmul(w_old))
		prox_G = np.minimum(polytope_lim, grad_F - L*w_old)
		y_new = w_old - (grad_F - prox_G)/L
		t_new = (1 + sqrt(1 + 4*t_old**2))/2  # this inertia term is what makes it fast
		w_new = y_new + (t_old - 1)/t_new*(y_new - y_old)
		x_new = point + polytope_mat.transpose_matmul(y_new)
		# save any points that are close enough to being in
		if np.all(polytope_mat@x_new <= polytope_lim + tolerance):
			candidates.append(x_new)
			# and terminate once we get enough of them
			if len(candidates) >= certainty:
				best = np.argmin(np.linalg.norm(np.array(candidates) - point, axis=1))
				return candidates[best]
		t_old, w_old, y_old = t_new, w_new, y_new
	return x_new
