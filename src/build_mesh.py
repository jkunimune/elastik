#!/usr/bin/env python
"""
build_mesh.py

take an interruption file, and use it to generate and save a basic interrupted map
projection mesh that can be further optimized.
all angles are in radians. indexing is z[i,j] = z(ф[i], λ[j])
"""
import queue

import h5py
import numpy as np
from matplotlib import pyplot as plt

from util import bin_index, bin_centers, wrap_angle, EARTH

# filename of the borders to use
SECTIONS_FILE = "../spec/cuts_mountains.txt"
# how many cells per 90°
RESOLUTION = 8#18
# filename of mesh at which to save it
MESH_FILE = "../spec/mesh_mountains.h5"
# locations of various straits that should be shown continuously
STRAITS = np.radians([(66.5, -169.0), # Bering
                      (9.1, -79.7), # Panama canal
                      (30.7, 32.3), # Suez Canal
                      (1.8, 102.4), # Malacca
                      (-9, 102.4), # Sunda (offset to make it aline with adjacent ones)
                      (-9, 114.4), # various indonesian straits
                      (-9, 142.4), # Torres (offset to make it aline with adjacent ones)
                      ])
# the distance around a strait that should be duplicated for clarity
STRAIT_RADIUS = 1800


class Section:
	def __init__(self, left_border: np.ndarray, rite_border: np.ndarray, glue_on_north: bool):
		""" a polygon that selects a portion of the globe bounded by two cuts originating
		    from the same point (the "cut_tripoint") and two continuous seams from those
		    cuts to a different point (the "glue_tripoint")
		    :param glue_on_north: whether the glue_tripoint should be the north pole (if
		                          not, it will be the south pole)
		"""
		if not (left_border[0, 0] == rite_border[0, 0] and left_border[0, 1] == rite_border[0, 1]):
			raise ValueError("the borders are supposed to start at the same point")

		self.cut_border = np.concatenate([left_border[::-1, :], rite_border])

		self.glue_border = Section.path_through_pole(self.cut_border[-1, :],
		                                             self.cut_border[0, :],
		                                             glue_on_north)

		self.border = np.concatenate([self.cut_border[:-1, :], self.glue_border])


	@staticmethod
	def path_through_pole(start: np.ndarray, end: np.ndarray, north: bool) -> np.ndarray:
		""" find a simple path that goes to the nearest pole, circles around it clockwise,
		    and then goes to the endpoint. assume the y axis to be periodic, and break the
		    path up at the antimeridian if necessary. the poles are at x = ±pi/2.
			:param start: the 2-vector at which to start
			:param end: the 2-vector at which to finish
			:param north: whether the nearest pole is north (vs south)
			:return: the n×2 path array
		"""
		sign = 1 if north else -1
		# start with some strait lines
		path = [start, [sign*np.pi/2, start[1]], [sign*np.pi/2, end[1]], end]
		# if it looks like it's circling the rong way
		if np.sign(start[1] - end[1]) != sign:
			path.insert(2, [sign*np.pi/2, -sign*np.pi])
			path.insert(3, [sign*np.pi/2,  sign*np.pi])
		# if the direction could still be considerd ambiguous
		for k in range(1, len(path)):
			dy = abs(path[k][1] - path[k - 1][1])
			if dy > np.pi and dy != 2*np.pi:
				path.insert(k, [sign*np.pi/2, 0])
				break
		return np.array(path)


	def inside(self, x_edges: np.ndarray, y_edges: np.ndarray) -> np.ndarray:
		""" find the locus of tiles binned by x and y that are inside this section. count
		    tiles that intersect the boundary as in
		    :param x_edges: the bin edges for axis 0
		    :param y_edges: the bin edges for axis 1
		    :return: a boolean grid of True for in and False for out
		"""
		# start by including anything on the border
		included = Section.cells_touched(x_edges, y_edges, self.border)

		# then do a simple polygon inclusion test
		x_centers = bin_centers(x_edges)
		for j in range(y_edges.size - 1):
			y = (y_edges[j] + y_edges[j + 1])/2
			x_crossings = []
			for k in range(self.border.shape[0] - 1): # check each segment
				x0, y0 = self.border[k, :]
				x1, y1 = self.border[k + 1, :]
				crosses = (y0 >= y) != (y1 >= y) # to see if it crosses this ray
				if abs(y1 - y0) > np.pi:
					crosses = not crosses # remember to account for wrapping
				if crosses:
					x_crossings.append(np.interp(y, [y0, y1], [x0, x1]))
			x_crossings = np.sort(x_crossings)
			num_crossings = np.sum(x_crossings[None, :] < x_centers[:, None], axis=1) # and apply the even/odd rule
			included[:, j] |= num_crossings%2 == 1

		return included


	def shared(self, x_edges: np.ndarray, y_edges: np.ndarray) -> np.ndarray:
		""" find the locus of tiles binned by x_edges and λ that span a soft glue border
		    between this section and another
		    :param x_edges: the bin edges for axis 0
		    :param y_edges: the bin edges for axis 1
		    :return: a boolean grid of True for shared and False for not shared
		"""
		return Section.cells_touched(x_edges, y_edges, self.glue_border)


	@staticmethod
	def cells_touched(x_edges: np.ndarray, y_edges: np.ndarray, path: np.ndarray) -> np.ndarray:
		""" find and mark each tile binned by x_edges and y_edges that is touched by this
	        polygon path
	        :param x_edges: the bin edges for axis 0
	        :param y_edges: the bin edges for axis 1
	        :param path: a n×2 array of ordered x and y coordinates
	        :return: a boolean grid of True for in and False for out
		"""
		touched = np.full((x_edges.size - 1, y_edges.size - 1), False)
		for i in range(path.shape[0] - 1):
			x0, y0 = path[i, :]
			x1, y1 = path[i + 1, :]
			touched[bin_index(x0, x_edges), bin_index(y0, y_edges)] = True
			if x0 != x1:
				i_crossings, j_crossings = Section.grid_intersections(x_edges, y_edges[:-1], x0, y0, x1, y1, False, True)
				touched[i_crossings, j_crossings] = True
				touched[i_crossings - 1, j_crossings] = True
			if y0 != y1:
				j_crossings, i_crossings = Section.grid_intersections(y_edges[:-1], x_edges, y0, x0, y1, x1, True, False)
				touched[i_crossings, j_crossings] = True
				touched[i_crossings, j_crossings - 1] = True
		return touched


	@staticmethod
	def grid_intersections(x_values: np.ndarray, y_edges: np.ndarray,
	                       x0: float, y0: float, x1: float, y1: float,
	                       periodic_x: bool, periodic_y: bool
	                       ) -> tuple[np.ndarray, np.ndarray]:
		""" bin the y value at each point where this single line segment crosses one of
		    the x_values.
	        :param x_values: the values at which we should detect and bin positions (must
		                     be evenly spaced if periodic)
		    :param y_edges: the edges of the bins in which to place y values (must be
		                    evenly spaced if periodic)
	        :param x0: the x coordinate of the start of the line segment
	        :param y0: the y coordinate of the start of the line segment
	        :param x1: the x coordinate of the end of the line segment
	        :param y1: the y coordinate of the end of the line segment
	        :param periodic_x: whether the x axis must be treated as periodic
	        :param periodic_y: whether the y axis must be treated as periodic
		    :return: the 1D array of x value indices and the 1D array of y bin indices
		"""
		# make sure we don't have to worry about periodicity issues
		if periodic_x and abs(x1 - x0) > np.pi:
			shift = bin_index(max(x0, x1), x_values)
			x_step = x_values[1] - x_values[0]
			i_crossings, j_crossings = Section.grid_intersections(
				x_values, y_edges,
				wrap_angle(x0 - x_step*shift), y0,
				wrap_angle(x1 - x_step*shift), y1,
				periodic_x, periodic_y)
			return (i_crossings + shift)%x_values.size, j_crossings
		elif periodic_y and abs(y0 - y1) > np.pi:
			shift = bin_index(max(y0, y1), y_edges)
			y_step = y_edges[1] - y_edges[0]
			i_crossings, j_crossings = Section.grid_intersections(
				x_values, y_edges,
				x0, wrap_angle(y0 - y_step*shift),
				x1, wrap_angle(y1 - y_step*shift),
				periodic_x, periodic_y)
			return i_crossings, (j_crossings + shift)%y_edges.size
		# and we want to be able to assume they go left to rite
		elif x1 < x0:
			return Section.grid_intersections(x_values, y_edges, x1, y1, x0, y0, periodic_x, periodic_y)
		elif x1 > x0:
			i0 = bin_index(x0, x_values) + 1
			i1 = bin_index(x1, x_values)
			i_crossings = np.arange(i0, i1 + 1)
			x_crossings = x_values[i_crossings]
			y_crossings = np.interp(x_crossings, [x0, x1], [y0, y1])
			j_crossings = bin_index(y_crossings, y_edges)
			return i_crossings, j_crossings
		else:
			return np.empty((0,), dtype=int), np.empty((0,), dtype=int)


def expand_bool_array(arr: np.ndarray) -> np.ndarray:
	""" create an array one bigger in both dimensions representing the anser to the
	    question: are any of the surrounding pixels True? """
	out = np.full((arr.shape[0] + 1, arr.shape[1] + 1), False)
	out[:-1, :-1] |= arr
	out[:-1, 1:] |= arr
	out[1:, :-1] |= arr
	out[1:, 1:] |= arr
	out[:, 0] |= out[:, -1] # don't forget to account for periodicity on axis 1
	out[:, -1] = out[:, 0]
	return out


def load_sections(filename: str) -> list[Section]:
	data = np.radians(np.loadtxt(filename))
	cut_tripoint = data[0, :]
	starts = np.nonzero((data[:, 0] == cut_tripoint[0]) & (data[:, 1] == cut_tripoint[1]))[0]
	cuts = []
	for l in range(starts.size):
		try:
			cuts.append(data[starts[l]:starts[l + 1]])
		except IndexError:
			cuts.append(data[starts[l]:])
	sections = []
	for l in range(len(cuts)):
		sections.append(Section(cuts[l], cuts[(l + 1)%len(cuts)], cut_tripoint[0] < 0))
	return sections


def save_mesh(filename: str, ф: np.ndarray, λ: np.ndarray, nodes: np.ndarray, sections: list[Section]) -> None:
	""" save the mesh for future use in a map projection HDF5
	    :param filename: the name of the file at which to save it
	    :param ф: the (m+1) array of latitudes positions at which there are nodes
	    :param λ: the (l+1) array of longitudes at which there are nodes
	    :param nodes: the (n × m+1 × l+1 × 2) array of projected cartesian coordinates (n
	                  is the number of sections)
	    :param sections: list of Sections, each corresponding to a layer of Cells and Nodes
	"""
	num_sections = len(sections)
	num_ф = ф.size - 1
	num_λ = λ.size - 1
	assert nodes.shape[0] == num_sections
	assert nodes.shape[1] == num_ф + 1
	assert nodes.shape[2] == num_λ + 1

	with h5py.File(filename, "w") as file:
		file.attrs["num_sections"] = num_sections
		for h in range(num_sections):
			dset = file.create_dataset(f"section{h}/latitude", shape=(num_ф + 1))
			dset[:] = np.degrees(ф)
			dset = file.create_dataset(f"section{h}/longitude", shape=(num_λ + 1))
			dset[:] = np.degrees(λ)
			dset = file.create_dataset(f"section{h}/projection", shape=(num_ф + 1, num_λ + 1, 2))
			dset[:, :, :] = nodes[h, :, :, :]
			dset = file.create_dataset(f"section{h}/border", shape=sections[h].border.shape)
			dset[:, :] = np.degrees(sections[h].border)


if __name__ == "__main__":
	# start by defining a grid of Cells
	ф = np.linspace(-np.pi/2, np.pi/2, 2*RESOLUTION + 1)
	dф = ф[1] - ф[0]
	num_ф = ф.size - 1
	λ = np.linspace(-np.pi, np.pi, 4*RESOLUTION + 1)
	dλ = λ[1] - λ[0]
	num_λ = λ.size - 1

	# load the interruptions
	sections = load_sections(SECTIONS_FILE)
	include_cells = np.full((len(sections), num_ф, num_λ), False)
	share_cells = np.full((num_ф, num_λ), False)
	for h, section in enumerate(sections):
		include_cells[h, :, :] = section.inside(ф, λ)
		share_cells[:, :] |= section.shared(ф, λ)

	for ф_strait, λ_strait in STRAITS:
		within_ф = abs(bin_centers(ф) - ф_strait) < STRAIT_RADIUS/EARTH.R
		within_λ = abs(wrap_angle(bin_centers(λ) - λ_strait)) < STRAIT_RADIUS/EARTH.R/np.cos(ф_strait)
		within = np.all(np.meshgrid(within_ф, within_λ, indexing="ij"), axis=0)
		include_cells[:, within] = True

	for h, section in enumerate(sections):
		plt.figure()
		plt.pcolormesh(λ, ф, np.where(include_cells[h, :, :], np.where(share_cells, 2, 1), 0))
		plt.plot(section.border[:, 1], section.border[:, 0], "k")
	plt.show()

	# change include and share to Node arrays instead of cell arrays
	share = expand_bool_array(share_cells)
	include = np.full((len(sections), num_ф + 1, num_λ + 1), False)
	for h in range(len(sections)):
		include[h, :, :] = expand_bool_array(include_cells[h, :, :])

	# create the node array
	nodes = np.full((len(sections), num_ф + 1, num_λ + 1, 2), np.nan)

	# now choose a longitude at which to seed it
	h_seed, j_seed = None, None
	for h in range(len(sections)):
		for j in range(num_λ): # there's bound to be at least one meridian that runs unbroken from pole to pole
			if np.all(include[h, :, j]):
				h_seed, j_seed = h, j
				break # it'll throw an index error if there is no such meridian

	# bild the cue from that seed
	cue = queue.Queue()
	for i in range(0, num_ф + 1):
		cue.put((h_seed, i, j_seed))
	# go out from the seed
	plt.figure()
	while not cue.empty():
		# take the next node that needs to be added
		h0, i, j0 = cue.get()
		if include[h0, i, j0] and np.isnan(nodes[h0, i, j0, 0]):
			# decide how many longitudes it spans
			if i == 0 or i == num_ф:
				all_js = np.arange(num_λ + 1)
			elif j0 == 0 or j0 == num_λ:
				all_js = [0, num_λ]
			else:
				all_js = [j0]
			# and in how many layers it exists
			if np.any(share[i, all_js]):
				all_hs = np.nonzero(np.any(include[:, i, all_js], axis=-1))[0] # this method of assining it to all hs is kind of janky, and may fail for more complicated section topologies
			else:
				all_hs = [h0]

			# now to determine its location!
			if h0 == h_seed and j0 == j_seed:
				# place the seed along the y axis
				node = [0, EARTH.R*ф[i]]
			else:
				# elsewhere, scan the surrounding nodes to get different suggestions for where it should go
				positions = []
				for h in all_hs:
					for j in all_js:
						for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
							ac = EARTH.R*dф if di != 0 else EARTH.R*dλ*np.cos(ф[i])
							for sign in [-1, 1]:
								i_a, j_a = i + di, j + dj
								i_b, j_b = i + di + sign*dj, j + dj - sign*di
								if i_a >= 0 and i_a <= num_ф and j_a >= 0 and j_a <= num_λ and \
										i_b >= 0 and i_b <= num_ф and j_b >= 0 and j_b <= num_λ:
									a = nodes[h, i_a, j_a]
									b = nodes[h, i_b, j_b]
								else:
									continue
								if not np.isnan(a[0]) and not np.isnan(b[0]):
									ab = np.hypot(b[0] - a[0], b[1] - a[1])
									if ab == 0:
										continue
									positions.append([
										a[0] - sign*ac/ab*(b[1] - a[1]),
										a[1] + sign*ac/ab*(b[0] - a[0]),
									])
				# it's possible that we need to come back to this later
				if len(positions) == 0:
					continue
				node = np.mean(positions, axis=0)

			# then decide on the location
			for h in all_hs:
				for j in all_js:
					nodes[h, i, j, :] = node

					# now check each of the node's neibors in a breadth-first manner
					for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
						i_next = i + di
						j_next = (j + dj + num_λ)%num_λ
						if i_next >= 0 and i_next < num_ф:
							if include[h, i_next, j_next] and np.isnan(nodes[h, i_next, j_next, 0]):
								cue.put((h, i_next, j_next))

		# show a status update
		if np.random.random() < 2e-2 or cue.empty():
			plt.clf()
			plt.scatter(*nodes.reshape((-1, 2)).T, s=5, color="k")
			for h in range(nodes.shape[0]):
				plt.plot(nodes[h, :, :, 0], nodes[h, :, :, 1], f"C{h}")
				plt.plot(nodes[h, :, :, 0].T, nodes[h, :, :, 1].T, f"C{h}")
			plt.axis("equal")
			plt.pause(.01)

	# save it to HDF5
	save_mesh(MESH_FILE, ф, λ, nodes, sections)

	plt.show()
