#!/usr/bin/env python
"""
find_drainage_divides.py

locate continental divides and save them in a way that can be used as optimal cuts for
the oceanic elastic map
all angles are in degrees. indexing is z[i,j] = z(ф[i], λ[j])
"""
from __future__ import annotations

import bisect
import os
from math import floor, ceil, nan, inf, copysign, sqrt
from typing import Iterable, Sequence

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import shapefile
import tifffile
from numpy.typing import NDArray

from util import bin_centers, bin_index, intersects, decimate_path

# how many pixels per degree
RESOLUTION = 10
# how to determine the height value of a pixel that contains multiple data points
REDUCTION = np.median
# what fraction of the found paths should be plotted
AMOUNT_TO_PLOT = 5e-2/RESOLUTION**2


def calculate_drainage_divides(endpoints: list[tuple[float, float]]):
	# define the allowd nodes of the path
	ф_map = np.linspace(-90, 90, int(180*RESOLUTION + 1))
	λ_map = np.linspace(-180, 180, int(360*RESOLUTION + 1))[:-1]

	# load the rivers
	rivers = load_river_data(ф_map, λ_map)

	# load the heightmap
	z_map = load_elevation_data(ф_map, λ_map)

	# search for paths between the endpoints
	tripoint: tuple[float, float] | None = None
	paths = []
	for i in range(1, len(endpoints)):
		start = endpoints[i]
		# first, draw a line from endpoint 1 to endpoint 0
		if len(paths) == 0:
			end = endpoints[0]
		# then, draw a line from endpoint 2 to any place on that path
		elif len(paths) == 1:
			end = paths[0]
		# after that, tripoint will be defined and you should always go to that
		else:
			end = tripoint
		print(f"finding path from {start} to {end}")
		paths.append(check_wrapping(
			find_hiest_path(start, end, ф_map, λ_map, z_map, rivers)))
		# remember to define the tripoint after the second path is drawn
		if len(paths) == 2:
			tripoint = paths[-1][-1, :]

	# adjust the first path to make it go to the tripoint like the others
	split_index = index_of_2d(tripoint, paths[0][:, 0], paths[0][:, 1])
	paths.append(paths[0][:split_index - 1:-1, :])
	paths[0] = paths[0][:split_index + 1, :]
	# then flip everything so that they all go away from the tripoint
	for i in range(len(paths)):
		paths[i] = paths[i][::-1, :]

	# next, remove any vertices on water
	for i in range(len(paths)):
		for j in range(len(paths[i]) - 1, -1, -1):
			if on_water(paths[i][j, :], ф_map, λ_map, z_map):
				paths[i] = np.concatenate([paths[i][:j, :], paths[i][j + 1:, :]])
	# finally, simplify all paths
	for i in range(len(paths)):
		paths[i] = decimate_path(paths[i], sqrt(2)/RESOLUTION)

	# save and plot them
	np.savetxt("../spec/cuts_mountains.txt", np.concatenate(paths), fmt="%.1f")  # type: ignore
	plt.figure()
	plot_map(z_map, (λ_map[0] - .5/RESOLUTION, λ_map[-1] + .5/RESOLUTION,
	                 ф_map[0] - .5/RESOLUTION, ф_map[-1] + .5/RESOLUTION))
	plt.contour(λ_map, ф_map, z_map, levels=np.linspace(0.5, 10000, 31), colors="w", linewidths=0.2)
	for river in rivers:
		i, j = np.transpose(river)
		plt.plot(λ_map[0] + (λ_map[1] - λ_map[0])*j,
		         ф_map[0] + (ф_map[1] - ф_map[0])*i, "#206", linewidth=0.5)
	for path in paths:
		plt.plot(path[:, 1], path[:, 0], "C3")
	plt.scatter([λ for ф, λ in endpoints], [ф for ф, λ in endpoints], c=f"C{len(paths)}")
	plt.xlabel("Longitude (°)")
	plt.ylabel("Latitude (°)")
	plt.tight_layout()
	plt.show()


def find_hiest_path(start: tuple[float, float], end: tuple[float, float] | NDArray[float],
                    x_nodes: NDArray[float], y_nodes: NDArray[float], z_nodes: NDArray[float],
                    barriers: list[list[tuple[float, float]]]) -> NDArray[float]:
	""" perform a dijkstra search to find the hiest path between start and end.  a path
	    may comprise a series of locally adjacent x values from x_nodes and y values from
	    y_nodes (diagonal steps are okay).  a path is defined as hier than another with
	    the same endpoints if its lowest point that it does not have in common with the
	    other is hier than the corresponding point on the other.  this means that such
	    paths will define watershed divides.
	    :param start: the start point of the search (must be in the node arrays)
	    :param end: the stop point or possible stop points of the search (each must be in
	                the node arrays)
	    :param x_nodes: a m-long array of allowed x positions
	    :param y_nodes: a n-long array of allowed y positions
	    :param z_nodes: a m×n array of the hite value at each pair of x and y
	    :param barriers: a set of paths that paths are not allowed to cross
	    :return: a l×2 array of the x and y coordinate at each point on the hiest path;
	             the 0th element will be start and the l-1th element will be end.
	"""
	if type(end[0]) is float:
		end = np.atleast_2d(end)
	# convert the start and end coordinates to indices
	i_start = round(index_of_1d(start[0], x_nodes))
	j_start = round(index_of_1d(start[1], y_nodes))
	i_ends = np.around(index_of_1d(end[:, 0], x_nodes)).astype(int)
	j_ends = np.around(index_of_1d(end[:, 1], y_nodes)).astype(int)
	visited = np.full(z_nodes.shape, False)

	# convert the barriers to an adjacency matrix
	adjacency = define_adjacency_matrix(x_nodes, y_nodes, barriers)

	# keep a list of the current paths in progress
	candidates: list[Path] = [Path([i_start], [j_start], [z_nodes[i_start, j_start]])]
	paths_to_plot: list[Path] = []
	while True:
		# take the most promising one
		path = candidates.pop()
		# if it reached a goal, we're all done here
		if index_of_2d(path.end, i_ends, j_ends) != -1:
			plt.close("all")
			return np.stack([x_nodes[path.i], y_nodes[path.j]], axis=-1)
		# otherwise, check that no one has beat it here
		i, j = path.end
		if not visited[i, j]:
			if np.random.random() < AMOUNT_TO_PLOT:
				paths_to_plot.append(path)
			# save it as a valid path
			visited[i, j] = True
			# and iterate thru all potential follow-ups
			for i_next in [i, i - 1, i + 1]:
				for j_next in [j, j - 1, j + 1]:
					j_next = (j_next + y_nodes.size)%y_nodes.size
					if adjacent((i, j), (i_next, j_next), adjacency):  # check if the step is valid
						if not visited[i_next, j_next]:  # and doesn’t cross any existing paths
							if path.len < 2 or not adjacent((i_next, j_next), (path.i[-2], path.j[-2]), adjacency): # and it must not form an unnecessary detour
								new_path = path + (i_next, j_next, z_nodes[i_next, j_next])
								bisect.insort(candidates, new_path)
		if len(paths_to_plot) == 6:
			plt.clf()
			for path in paths_to_plot:
				plt.plot(path.j, path.i, "--", linewidth=1.0, color="#000", zorder=10)
			plt.plot([path.end[1] for path in paths_to_plot],
			         [path.end[0] for path in paths_to_plot],
			         "o", markerfacecolor="#fff", markeredgecolor="#000", markersize=4, zorder=10)
			plot_map(z_nodes, (-0.5, y_nodes.size - 0.5, -0.5, x_nodes.size - 0.5))
			for barrier in barriers:
				ф, λ = zip(*barrier)
				plt.plot(λ, ф, color="#206",
				         linewidth=0.5, zorder=-2)
			i_nodes = np.arange(0, x_nodes.size)
			j_nodes = np.arange(0, y_nodes.size)
			plt.contour(j_nodes, i_nodes, np.where(visited, 0, 1),
			            levels=[0.5], colors="w",
			            linewidths=.5, zorder=-1)
			plt.axis([np.min(j_nodes[np.any(visited, axis=0)]),
			          np.max(j_nodes[np.any(visited, axis=0)]),
			          np.min(i_nodes[np.any(visited, axis=1)]),
			          np.max(i_nodes[np.any(visited, axis=1)])])
			plt.tight_layout()
			plt.pause(.01)
			paths_to_plot = []


def plot_map(z_nodes: NDArray[float], extent: tuple[float, float, float, float]):
	""" display the given elevation map as a nice looking map """
	plt.gca().set_facecolor("#205")
	plt.imshow(
		np.where(z_nodes > 0, z_nodes, nan),
		extent=extent,
		norm=colors.LogNorm(
			vmin=np.max(z_nodes)**-(1/5),
			vmax=np.max(z_nodes)),
		origin="lower", zorder=-3)


def load_elevation_data(ф_nodes: NDArray[float], λ_nodes: NDArray[float]) -> NDArray[float]:
	""" look for tiff files in the data/elevation/ folder and tile them together
	    to form a single global map, with its resolution set by ф_nodes and λ_nodes.
	    each pixel will correspond to one node and have value equal to the average
	    elevation in the region that is closer to it than to any other node, accounting
	    for λ periodicity
	    :param ф_nodes: the latitudes that our path is allowd to use for vertices
	    :param λ_nodes: the longitudes that our path is allowd to use for vertices
	"""
	# first establish the bin edges that will determine which pixel goes to which node
	ф_bins = np.concatenate([[-inf], bin_centers(ф_nodes), [inf]])
	λ_bins = np.concatenate([[-inf], bin_centers(λ_nodes), [(λ_nodes[0] + λ_nodes[-1])/2 + 180, inf]])

	# then begin bilding the map
	z_nodes = np.full((ф_nodes.size, λ_nodes.size), nan)

	# look at each data file (they may not achieve full coverage)
	for filename in os.listdir("../data/elevation"):
		print(f"loading {filename}")
		z_data = tifffile.imread(f"../data/elevation/{filename}")

		# read its location and assine node indices
		ф0 = float(filename[-6:-4]) * (1 if filename[-7] == "n" else -1)
		ф_data = bin_centers(np.linspace(ф0, ф0 - 50, z_data.shape[0] + 1))
		i_data = bin_index(ф_data, ф_bins)
		λ0 = float(filename[-10:-7]) * (1 if filename[-11] == "e" else -1)
		λ_data = bin_centers(np.linspace(λ0, λ0 + 40, z_data.shape[1] + 1))
		j_data = bin_index(λ_data, λ_bins)%z_nodes.shape[1]

		# iterate thru the touchd nodes and assine values
		for i in np.unique(i_data):
			for j in np.unique(j_data):
				j = j%z_nodes.shape[1]
				z_pixel = np.maximum(0, z_data[i_data==i][:, j_data==j])
				if np.isnan(z_nodes[i, j]):
					z_nodes[i, j] = REDUCTION(z_pixel)
				else:
					z_nodes[i, j] = REDUCTION([z_nodes[i, j], REDUCTION(z_pixel)])

	z_nodes[z_nodes < 0] = 0
	z_nodes[np.isnan(z_nodes)] = -inf
	return z_nodes


def load_river_data(ф_nodes: NDArray[float], λ_nodes: NDArray[float]) -> list[list[tuple[float, float]]]:
	""" load some river data.
	    :param ф_nodes: the latitudes that our path is allowd to use for vertices
	    :param λ_nodes: the longitudes that our path is allowd to use for vertices
	    :return: a list of rivers, where each river is a list of ф,λ points (degrees)
	"""
	# start by loading the rivers as lists
	rivers: list[list[tuple[float, float]]] = []
	with shapefile.Reader(f"../data/ne_50m_rivers_lake_centerlines.zip") as shape_f:
		for record, shape in zip(shape_f.records(), shape_f.shapes()):
			if "運河" in record.name_ja:
				continue  # make sure to skip all canals
			parts = np.concatenate([shape.parts, [len(shape.points)]])
			for k in range(1, len(parts)):
				points = shape.points[parts[k - 1]:parts[k]]
				if len(points) > 0:
					λ, ф = zip(*points)
					i = index_of_1d(ф, ф_nodes)
					j = index_of_1d(λ, λ_nodes)
					rivers.append(list(zip(i, j)))
	return rivers


def define_adjacency_matrix(ф_nodes: NDArray[float], λ_nodes: NDArray[float],
                            barriers: Iterable[Sequence[tuple[float, float]]]) -> NDArray[bool]:
	""" generate an adjacency graph given some barriers expressed as polylines
	    :param ф_nodes: the latitudes that our path is allowd to use for vertices
	    :param λ_nodes: the longitudes that our path is allowd to use for vertices
	    :param barriers: a set of polylines that separate nearby nodes from each other
	    :return: an m×n×3×3 array where n is the number of latitudes, n is the number of longitudes, and
	             matrix[i, j, di, dj] tells you whether a valid drainage divide can step from point i j to the point
	             <di, dj> away (it’s set up to be indexed with di and dj in the set [-1, 0, 1], *not* with [0, 1, 2]).
	"""
	adjacency = np.full((ф_nodes.size, λ_nodes.size, 3, 3), True)
	adjacency[:, :, 0, 0] = False  # don’t allow a step from one point to itself
	adjacency[0, :, -1, :] = False  # don’t allow it to step south of the south pole
	adjacency[-1, :, 1, :] = False  # don’t allow it to step north of the north pole

	for barrier in barriers:
		# for each river segment
		for k in range(1, len(barrier)):
			a = barrier[k - 1]
			b = barrier[k]
			# find which indices may be affected
			i_min = floor(min(a[0], b[0]))
			i_max = ceil(max(a[0], b[0]))
			j_min = floor(min(a[1], b[1]))
			j_max = ceil(max(a[1], b[1]))
			for i in range(i_min, i_max + 1):
				for j in range(j_min, j_max + 1):
					for di in [-1, 0, 1]:
						for dj in [-1, 0, 1]:
							if intersects(a, b, (i, j), (i + di, j + dj)):
								adjacency[i, j, di, dj] = False
								adjacency[i + di, j + dj, -di, -dj] = False
	return adjacency


def on_water(point: tuple[float, float], ф_map: NDArray[float], λ_map: NDArray[float], z_map: NDArray[float]) -> bool:
	nearest_i = np.argmin(abs(point[0] - ф_map))
	nearest_j = np.argmin(abs(point[1] - λ_map))
	return z_map[nearest_i, nearest_j] <= 0


def check_wrapping(points: NDArray[float]) -> NDArray[float]:
	""" find any segments that look like they wrap periodically and make them more
	    explicit, assuming that any such crossing will have one point at y=-180
	"""
	points = list(points)
	for k in range(len(points) - 1, -1, -1):
		if points[k][1] == -180:
			if k + 1 < len(points) and points[k + 1][1] > 0:
				points.insert(k + 1, [points[k][0], 180])
			elif k - 1 >= 0 and points[k - 1][1] > 0:
				points.insert(k, [points[k][0], 180])
	return np.array(points)


def index_of_1d(x: float | NDArray[float], arr: NDArray[float]) -> float | NDArray[float]:
	""" find the fractional index for arr that would yields the value x """
	return np.interp(x, arr, np.arange(arr.size))


def index_of_2d(pair: tuple, x: NDArray[float], y: NDArray[float]) -> int:
	""" find the first index of x and y such that pair[0] == x[i] and paint[1] == y[i] exactly """
	assert len(pair) == 2
	if np.any((pair[0] == x) & (pair[1] == y)):
		return np.nonzero((pair[0] == x) & (pair[1] == y))[0][0]
	else:
		return -1


def adjacent(a: tuple[int, int], b: tuple[int, int], adjacency: NDArray[bool]) -> bool:
	""" are these two index pairs adjacent (diagonals count, and check for wraparound)? """
	i, j = a
	di = b[0] - a[0]
	dj = b[1] - a[1]
	if abs(dj) == adjacency.shape[1] - 1:
		dj = int(copysign(1, -dj))
	return abs(di) <= 1 and abs(dj) <= 1 and adjacency[i, j, di, dj]


class Path:
	def __init__(self, i: list[int], j: list[int], z_sorted: list[float], num_kinks: int = 0):
		""" a class that keeps track of a path thru a grid in a manner that can be easily sorted. """
		self.i = i # the x indices that define this path
		self.j = j # the y indices that define this path
		self.z_sorted = z_sorted # the sorted z values that rate this path
		self.len = len(i)
		assert self.len >= 1
		self.start = (i[0], j[0])
		self.end = (i[-1], j[-1])
		self.num_kinks = num_kinks

	def __add__(self, other: tuple[int, int, float]):
		i, j, z = other
		new_i = self.i + [i]
		new_j = self.j + [j]
		new_z_sorted = list(self.z_sorted)
		bisect.insort(new_z_sorted, z)
		new_num_kinks = self.num_kinks
		if self.len >= 2:
			if (i - self.i[-1] != self.i[-1] - self.i[-2]) or \
			   (j - self.j[-1] != self.j[-1] - self.j[-2]):
				new_num_kinks += 1
		return Path(new_i, new_j, new_z_sorted, new_num_kinks)

	def __lt__(self, other: Path):
		if self.z_sorted < other.z_sorted:
			return True
		elif self.z_sorted > other.z_sorted:
			return False
		else:
			return self.num_kinks > other.num_kinks


if __name__ == "__main__":
	calculate_drainage_divides(
		endpoints=[(-29.47, 29.27), (-49.02, -73.50), (-36.46, 148.26)]
	)
