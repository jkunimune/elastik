"""
optimize_cuts.py

use simulated annealing to locate continental divides and save them in a way that can be
used as optimal cuts for the oceanic map
"""
import bisect

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import os
import tifffile


# what fraction of the found paths should be plotted
AMOUNT_TO_PLOT = 5e-4
# how many pixels per degree
RESOLUTION = 4
# how to determine the value of a pixel that contains multiple data points
REDUCTION = np.mean


class Path:
	def __init__(self, i: list[int], j: list[int], hitemap: np.ndarray):
		""" a class that keeps track of a path thru a grid in a manner that can be easily
			sorted.
		"""
		self.i = i # the x indices that define this path
		self.j = j # the y indices that define this path
		self.z_sorted = sorted(hitemap[i, j]) # the sorted z values that rate this path
		self.start = (i[0], j[0])
		self.end = (i[-1], j[-1])


	def prune(self, hitemap: np.ndarray) -> None:
		""" edit this path in place to remove any node that is hier than both of its
		    neibors if its neibors are adjacent to each other.
		"""
		for k in range(len(self.i) - 2, 0, -1):
			di = abs(self.i[k + 1] - self.i[k - 1])
			dj = abs(self.j[k + 1] - self.j[k - 1])
			a = hitemap[self.i[k - 1], self.j[k - 1]]
			b = hitemap[self.i[k], self.j[k]]
			c = hitemap[self.i[k + 1], self.j[k + 1]]
			if di <= 1 and dj <= 1 and b > a and b > c:
				self.i.pop(k)
				self.j.pop(k)
				self.z_sorted.remove(b)


def bin(x: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
	""" I dislike the way numpy defines this function """
	return np.digitize(x, bin_edges) - 1


def bin_centers(bin_edges: np.ndarray) -> np.ndarray:
	""" calculate the center of each bin """
	return (bin_edges[1:] + bin_edges[:-1])/2


def round_index(x: float, arr: np.ndarray) -> int:
	""" find the index for arr that yields the value nearest x """
	return int(np.round(np.interp(x, arr, np.arange(arr.size))))


def load_elevation_data(ф_nodes: np.ndarray, λ_nodes: np.ndarray) -> np.array:
	""" look for tiff files in the data/elevation/ folder and tile them together
	    to form a single global map, with its resolution set by ф_nodes and λ_nodes.
	    each pixel will correspond to one node and have value equal to the average
	    elevation in the region that is closer to it than to any other node, accounting
	    for λ periodicity
	"""
	# first establish the bin edges that will determine which pixel goes to which node
	ф_bins = np.concatenate([[-np.inf], bin_centers(ф_nodes), [np.inf]])
	λ_bins = np.concatenate([[-np.inf], bin_centers(λ_nodes), [(λ_nodes[0] + λ_nodes[-1])/2 + 180, np.inf]])

	# then begin bilding the map
	z_nodes = np.full((ф_nodes.size, λ_nodes.size), np.nan)

	# look at each data file (they may not achieve full coverage)
	for filename in os.listdir("../data/elevation"):
		print(f"loading {filename}")
		z_data = tifffile.imread(f"../data/elevation/{filename}")

		# read its location and assine node indices
		ф0 = float(filename[-6:-4]) * (1 if filename[-7] == "n" else -1)
		ф_data = bin_centers(np.linspace(ф0, ф0 - 50, z_data.shape[0] + 1))
		i_data = bin(ф_data, ф_bins)
		λ0 = float(filename[-10:-7]) * (1 if filename[-11] == "e" else -1)
		λ_data = bin_centers(np.linspace(λ0, λ0 + 40, z_data.shape[1] + 1))
		j_data = bin(λ_data, λ_bins)%z_nodes.shape[1]

		# iterate thru the touchd nodes and assine values
		for i in np.unique(i_data):
			for j in np.unique(j_data):
				j = j%z_nodes.shape[1]
				z_pixel = np.maximum(0, z_data[i_data==i][:, j_data==j])
				if np.isnan(z_nodes[i, j]):
					z_nodes[i, j] = REDUCTION(z_pixel)
				else:
					z_nodes[i, j] = REDUCTION([z_nodes[i, j], REDUCTION(z_pixel)])

	z_nodes[(~np.isfinite(z_nodes)) | (z_nodes < 0)] = 0
	return z_nodes


def find_hiest_path(start: tuple[float, float], end: tuple[float, float],
                    x_nodes: np.ndarray, y_nodes: np.ndarray, z_nodes: np.ndarray
                    ) -> np.ndarray:
	""" perform a dijkstra search to find the hiest path between start and end.  a path
	    may comprise a series of locally adjacent x values from x_nodes and y values from
	    y_nodes (diagonal steps are okay).  a path is defined as hier than another with
	    the same endpoints if its lowest point that it does not have in common with the
	    other is hier than the corresponding point on the other.  this means that such
	    paths will define watershed divides.
	    :param: start the start point of the search (must be in the node arrays)
	    :param: end the stop point of the search (must be in the node arrays)
	    :param: x_nodes a n-long array of allowed x positions
	    :param: y_nodes a m-long array of allowed y positions
	    :param: z_nodes a n×m array of the hite value at each pair of x and y
	    :return: a l×2 array of the x and y coordinate at each point on the hiest path;
	             the 0th element will be start and the l-1th element will be end.
	"""
	i_start = round_index(start[0], x_nodes)
	j_start = round_index(start[1], y_nodes)
	i_end = round_index(end[0], x_nodes)
	j_end = round_index(end[1], y_nodes)
	visited = np.full(z_nodes.shape, False)

	# keep a list of the current paths in progress
	candidates: list[Path] = [Path([i_start], [j_start], z_nodes)]
	paths_to_plot: list[Path] = []
	while True:
		# take the most promising one
		path = candidates.pop()
		# if it reached the goal, we're all done here
		if path.end[0] == i_end and path.end[1] == j_end:
			path.prune(z_nodes)
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
			for di in [-1, 0, 1]:
				for dj in [-1, 0, 1]:
					if di != 0 or dj != 0:
						i_next = i + di
						j_next = (j + dj + y_nodes.size)%y_nodes.size
						if i_next > 0 and i_next < x_nodes.size:
							if not visited[i_next, j_next]:
								new_path = Path(path.i + [i_next],
								                path.j + [j_next], z_nodes)
								bisect.insort(candidates, new_path,
								              key=lambda path: path.z_sorted)
		if len(paths_to_plot) == 6:
			plt.clf()
			for path in paths_to_plot:
				plt.plot(path.j, path.i, "--")
				plt.scatter([path.start[1], path.end[1]], [path.start[0], path.end[0]])
			plt.autoscale(False)
			i_edges = np.arange(-0.5, x_nodes.size)
			j_edges = np.arange(-0.5, y_nodes.size)
			plt.pcolormesh(j_edges, i_edges, z_nodes, norm=colors.LogNorm(), zorder=-2)
			plt.contour(bin_centers(j_edges), bin_centers(i_edges), np.where(visited, 0, 1), levels=[0.5], colors="C6", linewidths=1, zorder=-1)
			plt.tight_layout()
			plt.pause(.01)
			paths_to_plot = []


if __name__ == "__main__":
	# define the start and end locations of the cuts
	endpoints = [(-36.46, 148.26), (-29.47, 29.27), (-46.60, -73.35)]

	# define the allowd nodes of the path
	ф_map = np.linspace(-90, 90, int(180*RESOLUTION + 1))
	λ_map = np.linspace(-180, 180, int(360*RESOLUTION + 1))[:-1]

	z_map = load_elevation_data(ф_map, λ_map)

	# search for paths between the endpoints
	paths = []
	for i in range(len(endpoints)):
		print(f"finding path from {endpoints[i]} to {endpoints[(i + 1)%len(endpoints)]}")
		paths.append(find_hiest_path(endpoints[i], endpoints[(i + 1)%len(endpoints)], ф_map, λ_map, z_map))

	# save and plot them
	np.savetxt("../spec/continental_divides.txt", np.concatenate(paths), fmt="%.1f")
	plt.figure()
	plt.contourf(λ_map, ф_map, z_map, levels=np.linspace(0.5, 10000, 21), vmax=3000)
	plt.contour(λ_map, ф_map, z_map, levels=np.linspace(0.5, 10000, 21), colors="k", linewidths=0.2)
	for path in paths:
		plt.plot(path[:, 1], path[:, 0])
	plt.scatter([λ for ф, λ in endpoints], [ф for ф, λ in endpoints], c=f"C{len(paths)}")
	plt.xlabel("Longitude (°)")
	plt.ylabel("Latitude (°)")
	plt.tight_layout()
	plt.show()
