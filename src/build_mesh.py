#!/usr/bin/env python
"""
build_mesh.py

take an interruption file, and use it to generate and save a basic interrupted map
projection mesh that can be further optimized.
all angles are in radians. indexing is z[i,j] = z(ф[i], λ[j])
"""
from math import cos, hypot, pi, ceil, sin, nan, tan, inf, copysign

import h5py
import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray

from util import bin_index, bin_centers, wrap_angle, EARTH, inside_region, interp


# locations of various straits that should be shown continuously
STRAITS = np.radians([(66.5, -169.0), # Bering
                      (9.1, -79.7), # Panama canal
                      (30.7, 32.3), # Suez Canal
                      (1.8, 102.4), # Malacca
                      (-10, 102.4), # Sunda (offset to make it aline with adjacent ones)
                      (-10, 116), # Lombok (offset to make it aline with adjacent ones)
                      (-10, 129), # Timor sea
                      (-60, -65), # Drake
                      ])
# the distance around a strait that should be duplicated for clarity
STRAIT_RADIUS = 1800/EARTH.R # (radians)


class Section:
	def __init__(self, left_border: NDArray[float], rite_border: NDArray[float], glue_tripoint: NDArray[float]):
		""" a polygon that selects a portion of the globe bounded by two cuts originating
		    from the same point (the "cut_tripoint") and two continuous seams from those
		    cuts to a different point (the "glue_tripoint")
		    :param glue_tripoint: the location at which the sections all meet and join together
		"""
		if not (left_border[0, 0] == rite_border[0, 0] and left_border[0, 1] == rite_border[0, 1]):
			raise ValueError("the borders are supposed to start at the same point")

		self.cut_border = np.concatenate([left_border[:0:-1, :], rite_border])
		self.glue_tripoint = glue_tripoint

		self.glue_border = Section.path_through(self.cut_border[-1, :],
		                                        self.glue_tripoint,
		                                        self.cut_border[0, :])

		self.border = np.concatenate([self.cut_border[:-1, :], self.glue_border])


	@staticmethod
	def path_through(start: NDArray[float], middle: NDArray[float], end: NDArray[float]) -> NDArray[float]:
		""" find a simple path that goes to the nearest pole, circles around it clockwise,
		    and then goes to the endpoint. assume the y axis to be periodic, and break the
		    path up at the antimeridian if necessary. the poles are at x = ±pi/2.
			:param start: the 2-vector at which to start
			:param middle: the 2-vector thru which to pass
			:param end: the 2-vector at which to finish
			:return: the n×2 path array
		"""
		# for normal points...
		if abs(middle[0]) < pi/2:
			return np.array([start, middle, end])
		# if the midpoint is a pole...
		else:
			sign = copysign(1, middle[0])
			# start with some strait lines
			path = [start, [sign*pi/2, start[1]], [sign*pi/2, end[1]], end]
			# if it looks like it's circling the rong way
			if np.sign(start[1] - end[1]) != sign:
				path.insert(2, [sign*pi/2, -sign*pi])
				path.insert(3, [sign*pi/2, sign*pi])
			for k in range(len(path) - 1, 0, -1):
				dy = abs(path[k][1] - path[k - 1][1])
				# if at any point the direction could still be considerd ambiguous, clarify it
				if dy > pi and dy != 2*pi:
					path.insert(k, [sign*pi/2, 0])
				# also, if there are any zero-length segments, remove them
				elif np.all(path[k] == path[k - 1]):
					path.pop(k)
			return np.array(path)


	def inside(self, x_edges: NDArray[float], y_edges: NDArray[float]) -> NDArray[bool]:
		""" find the locus of tiles binned by x and y that are inside this section. count
		    tiles that intersect the boundary as in
		    :param x_edges: the bin edges for axis 0
		    :param y_edges: the bin edges for axis 1
		    :return: a boolean grid of True for in and False for out
		"""
		# start by including anything on the border
		included = Section.cells_touched(x_edges, y_edges, self.border)

		# then do a simple polygon inclusion test
		included |= inside_region(bin_centers(x_edges), bin_centers(y_edges), self.border, period=2*pi)

		return included


	def shared(self, x_edges: NDArray[float], y_edges: NDArray[float]) -> NDArray[bool]:
		""" find the locus of tiles binned by x_edges and λ that span a soft glue border
		    between this section and another
		    :param x_edges: the bin edges for axis 0
		    :param y_edges: the bin edges for axis 1
		    :return: a boolean grid of True for shared and False for not shared
		"""
		# share any cells that are on the glue border
		shared = Section.cells_touched(x_edges, y_edges, self.glue_border)
		# except for this one weerd edge case with the poles
		for x_endpoint, y_endpoint in self.cut_border[[0, -1], :]:
			j = bin_index(y_endpoint, y_edges)
			if x_endpoint == -pi/2:
				shared[0, j - 1: j + 2] = False
			elif x_endpoint == pi/2:
				shared[-1, j - 1: j + 2] = False
		# and this other edge case with cuts that pass in and out of the shared region
		i = bin_index(self.cut_border[:, 0], x_edges)
		j = bin_index(self.cut_border[:, 1], y_edges)
		shared_points = shared[i, j]
		emerging = np.nonzero(shared_points[:-1] & ~shared_points[1:])[0]
		entering = np.nonzero(~shared_points[:-1] & shared_points[1:])[0] + 1
		shared[i[emerging[1:]], j[emerging[1:]]] = False # (strike all re-exiting cells)
		shared[i[entering[:-1]], j[entering[:-1]]] = False # (and all early-entering cells)

		return shared


	def choose_center(self):
		""" calculate the point that should go at the center of the stereographic
		    projection that minimizes the maximum distortion of this region of the globe
		"""
		ф_grid = np.linspace(-pi/2, pi/2, 13)
		λ_grid = np.linspace(-pi, pi, 25)
		ф_sample, λ_sample = bin_centers(ф_grid), bin_centers(λ_grid)
		inside = self.inside(ф_grid, λ_grid)
		max_distortion = np.where(inside, inf, 0)
		for points, importance in [(self.border, 1), (self.glue_border, 2)]:
			distance, _ = rotated_coordinates(
				ф_sample[:, np.newaxis, np.newaxis],
				λ_sample[np.newaxis, :, np.newaxis],
				points[np.newaxis, np.newaxis, :, 0],
				points[np.newaxis, np.newaxis, :, 1])
			distortion = importance/np.sin(distance/2)**2
			max_distortion = np.maximum(max_distortion, np.max(distortion, axis=2))
		best_i, best_j = np.unravel_index(np.argmin(max_distortion), max_distortion.shape)
		ф_anti, λ_anti = ф_sample[best_i], λ_sample[best_j]
		return -ф_anti, wrap_angle(λ_anti + pi)


	@staticmethod
	def cells_touched(x_edges: NDArray[float], y_edges: NDArray[float],
	                  path: NDArray[float]) -> NDArray[bool]:
		""" find and mark each tile binned by x_edges and y_edges that is touched by this
	        polygon path
	        :param x_edges: the bin edges for axis 0
	        :param y_edges: the bin edges for axis 1
	        :param path: a n×2 array of ordered x and y coordinates
	        :return: a boolean grid of True for in and False for out
		"""
		touched = np.full((x_edges.size - 1, y_edges.size - 1), False)
		for i in range(path.shape[0] - 1):
			x0, y0 = path[i, :] + 1e-6 # add a tiny amount so there's less roundoff instability
			x1, y1 = path[i + 1, :] + 1e-6
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
	def grid_intersections(x_values: NDArray[float], y_edges: NDArray[float],
	                       x0: float, y0: float, x1: float, y1: float,
	                       periodic_x: bool, periodic_y: bool
	                       ) -> tuple[NDArray[int], NDArray[int]]:
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
		if periodic_x and abs(x1 - x0) > pi:
			shift = bin_index(max(x0, x1), x_values)
			x_step = x_values[1] - x_values[0]
			i_crossings, j_crossings = Section.grid_intersections(
				x_values, y_edges,
				wrap_angle(x0 - x_step*shift), y0,
				wrap_angle(x1 - x_step*shift), y1,
				periodic_x, periodic_y)
			return (i_crossings + shift)%x_values.size, j_crossings
		elif periodic_y and abs(y0 - y1) > pi:
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
			y_crossings = interp(x_crossings, x0, x1, y0, y1)
			j_crossings = bin_index(y_crossings, y_edges)
			return i_crossings, j_crossings
		else:
			return np.empty((0,), dtype=int), np.empty((0,), dtype=int)


def expand_bool_array(arr: NDArray[bool]) -> NDArray[bool]:
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


def rotated_coordinates(ф_ref: NDArray[float], λ_ref: NDArray[float],
                        ф1: NDArray[float], λ1: NDArray[float]) -> tuple[NDArray[float], NDArray[float]]:
	""" return the polar distance and longitude relative to an oblique reference pole """
	x_rotate = np.sin(ф_ref)*np.cos(ф1)*np.cos(λ1 - λ_ref) - np.cos(ф_ref)*np.sin(ф1)
	y_rotate = np.cos(ф1)*np.sin(λ1 - λ_ref)
	z_rotate = np.cos(ф_ref)*np.cos(ф1)*np.cos(λ1 - λ_ref) + np.sin(ф_ref)*np.sin(ф1)
	p_rotate = pi/2 - np.arctan(z_rotate/np.hypot(x_rotate, y_rotate))
	λ_rotate = np.arctan2(y_rotate, x_rotate)
	return p_rotate, λ_rotate


def resolve_path(фs: NDArray[float], λs: NDArray[float],
                 resolution: float) -> tuple[NDArray[float], NDArray[float]]:
	""" refire a path such that its segments are no longer than resolution """
	assert фs.size == λs.size
	new_фs, new_λs = [фs[0]], [λs[0]]
	for i in range(1, фs.size):
		if abs(λs[i] - λs[i - 1]) <= pi:
			distance = hypot(фs[i] - фs[i - 1], λs[i] - λs[i - 1])
			segment_points = np.linspace(0, 1, ceil(distance/resolution) + 1)[1:]
			for t in segment_points:
				new_фs.append((1 - t)*фs[i - 1] + t*фs[i])
				new_λs.append((1 - t)*λs[i - 1] + t*λs[i])
	return np.array(new_фs), np.array(new_λs)


def load_sections(filename: str) -> list[Section]:
	""" load a cuts_*.txt file and convert it into a list of Sections """
	data = np.radians(np.loadtxt(filename))
	glue_tripoint = data[0, :]
	cut_tripoint = data[1, :]
	starts, = np.nonzero(np.all(data == cut_tripoint, axis=1))
	endpoints = np.concatenate([starts, [None]])
	cuts = []
	for h in range(starts.size):
		cuts.append(data[endpoints[h]:endpoints[h + 1]])
	sections = []
	for h in range(len(cuts)):
		sections.append(Section(cuts[h - 1], cuts[h], glue_tripoint))
	return sections


def save_mesh(filename: str, ф: NDArray[float], λ: NDArray[float],
              nodes: NDArray[float], sections: list[Section]) -> None:
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
			file.create_dataset(f"section{h}/latitude", data=np.degrees(ф))
			file.create_dataset(f"section{h}/longitude", data=np.degrees(λ))
			file.create_dataset(f"section{h}/projection", data=nodes[h, :, :, :])
			file.create_dataset(f"section{h}/border", data=np.degrees(sections[h].border))


def build_mesh(name: str, resolution: int):
	""" bild a mesh
	    :param name: "basic" | "oceans" | "mountains"
	    :param resolution: how many cells per 90°
	"""
	# start by defining a grid of Cells
	ф = np.linspace(-pi/2, pi/2, 2*resolution) # note that this is intentionally odd
	num_ф = ф.size - 1
	λ = np.linspace(-pi, pi, 4*resolution)
	num_λ = λ.size - 1

	# load the interruptions
	sections = load_sections(f"../spec/cuts_{name}.txt")

	# create the node array
	nodes = np.full((len(sections), num_ф + 1, num_λ + 1, 2), nan)
	include_nodes = np.full((len(sections), num_ф + 1, num_λ + 1), False)
	share_cells = np.full((num_ф, num_λ), False)

	# for each section
	for h, section in enumerate(sections):
		# get the main bitmaps of merit from its border
		share_cells |= section.shared(ф, λ)
		include_cells = section.inside(ф, λ)

		# add in any straits that happen to be split across it's edge
		ф_border, λ_border = resolve_path(section.cut_border[:, 0], section.cut_border[:, 1], STRAIT_RADIUS)
		for ф_strait, λ_strait in STRAITS:
			border_near_strait =\
				(abs(ф_border - ф_strait) < STRAIT_RADIUS/2) &\
				(abs(wrap_angle(λ_border - λ_strait)) < STRAIT_RADIUS/2/cos(ф_strait))
			if np.any(border_near_strait):
				ф_grid = bin_centers(ф)[:, np.newaxis]
				λ_grid = bin_centers(λ)[np.newaxis, :]
				cell_near_strait = \
					(abs(ф_grid - ф_strait) < STRAIT_RADIUS) & \
					(abs(wrap_angle(λ_grid - λ_strait)) < STRAIT_RADIUS/cos(ф_strait))
				include_cells[cell_near_strait] = True

		# force it to include the whole polar region when it touches the polar region
		for i_pole in [0, -1]:
			if np.any(include_cells[i_pole, :]):
				include_cells[i_pole, :] = True
			# and share the whole pole when some of the pole is shared
			if np.any(share_cells[i_pole, :]):
				share_cells[i_pole, :] = True

		include_nodes[h, :, :] = expand_bool_array(include_cells)

		# and create an oblique stereographic projection just for it
		ф_center, λ_center = section.choose_center()
		p_transform, λ_transform = rotated_coordinates(
			ф_center, λ_center, ф[:, np.newaxis], λ[np.newaxis, :])
		r, θ = np.tan(p_transform/2), λ_transform
		# shift it so the shared point is at the origin for all sections
		p_gluepoint, λ_gluepoint = rotated_coordinates(
			ф_center, λ_center, section.glue_tripoint[0], section.glue_tripoint[1])
		r_gluepoint, θ_gluepoint = tan(p_gluepoint/2), λ_gluepoint
		x1 =  r*np.sin(θ) - r_gluepoint*np.sin(θ_gluepoint)
		y1 = -r*np.cos(θ) + r_gluepoint*np.cos(θ_gluepoint)
		# rotate and scale it so it's locally continuus at the shared point
		_, β_center = rotated_coordinates(
			section.glue_tripoint[0], section.glue_tripoint[1], ф_center, λ_center)
		scale = 3*EARTH.R*cos(p_gluepoint/2)**2
		rotation = β_center - θ_gluepoint - pi
		x2 = scale*(x1*cos(rotation) - y1*sin(rotation))
		y2 = scale*(x1*sin(rotation) + y1*cos(rotation))
		nodes[h, include_nodes[h, :, :], 0] = x2[include_nodes[h, :, :]]
		nodes[h, include_nodes[h, :, :], 1] = y2[include_nodes[h, :, :]]

		# plot it
		plt.figure(f"{name.capitalize()} mesh, section {h}")
		plt.pcolormesh(λ, ф, np.where(include_cells, np.where(share_cells, 2, 1), 0))
		plt.plot(section.border[:, 1], section.border[:, 0], "k")
		plt.scatter(section.cut_border[[0, -1], 1], section.cut_border[[0, -1], 0], c="k", s=20)
		plt.scatter(λ_center, ф_center, c="k", s=50)

	share_nodes = expand_bool_array(share_cells)

	# finally, blend the sections together at their boundaries
	mean_nodes = np.tile(np.nanmean(nodes, axis=0), (len(sections), 1, 1, 1))
	nodes[include_nodes & share_nodes, :] = mean_nodes[include_nodes & share_nodes, :]
	# and assert the identity of the poles and antimeridian
	node_exists = np.all(np.isfinite(nodes), axis=3)
	for h in range(nodes.shape[0]):
		for i_pole in [0, -1]:
			if np.any(np.isfinite(nodes[h, i_pole])):
				nodes[h, i_pole, :, :] = np.nanmean(nodes[h, i_pole, :, :], axis=0)
	nodes[:, :, -1, :] = nodes[:, :, 0, :]
	nodes[~node_exists, :] = nan  # but make sure nan nodes stay nan

	# show the result
	plt.figure(f"{name.capitalize()} mesh")
	plt.scatter(*nodes.reshape((-1, 2)).T, s=5, color="k")
	for h in range(nodes.shape[0]):
		plt.plot(nodes[h, :, :, 0], nodes[h, :, :, 1], f"C{h}", linewidth=1)
		plt.plot(nodes[h, :, :, 0].T, nodes[h, :, :, 1].T, f"C{h}", linewidth=1)
	plt.axis("equal")

	# save it to HDF5
	save_mesh(f"../spec/mesh_{name}.h5", ф, λ, nodes, sections)


if __name__ == "__main__":
	build_mesh("basic", 25)
	# build_mesh("oceans", 25)
	# build_mesh("mountains", 25)
	plt.show()
