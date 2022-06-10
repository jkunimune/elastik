#!/usr/bin/env python
"""
build_mesh.py

take an interruption file, and use it to generate and save a basic interrupted map
projection mesh that can be further optimized.
all angles are in radians. indexing is z[i,j] = z(ф[i], λ[j])
"""
import queue

import numpy as np
from matplotlib import pyplot as plt

from util import bin_index, bin_centers, wrap_angle

# filename of the borders to use
SECTIONS_FILE = "../spec/cuts_mountains.txt"
# how many cells per 90°
RESOLUTION = 7
# radius of earth in km
EARTH_RADIUS = 6370


class Node:
	def __init__(self, i: int, j: int, x: float, y: float):
		""" a point at which the relation between (ф, λ) and (x, y) is fixed """
		self.i: int = i
		self.j: int = j
		self.x: float = x
		self.y: float = y


	def __repr__(self) -> str:
		return f"Node({self.i}, {self.j}, {self.x:.0f}, {self.y:.0f})"


class Cell:
	def __init__(self, i_sw: int, j_sw: int, max_j: int,
	             sw: Node or None, se: Node or None,
	             ne: Node or None, nw: Node or None,
	             top: float, bottom: float, hite: float):
		""" a trapezoidal region formed by four adjacent Nodes. it keeps track of the
		    canonical lengths of each of its sides so that it can calculate its own
		    derivatives. the constructor will create and place new Nodes if some are None.
		    :param: i_sw the ф index of the southwestern Node
		    :param: j_sw the λ index of the southwestern Node
		    :param: sw the southwestern Node to use or None if we need to create one
		    :param: se the southeastern Node to use or None if we need to create one
		    :param: ne the northeastern Node to use or None if we need to create one
		    :param: nw the northwestern Node to use or None if we need to create one
		    :param: top the length of the northern border (forgive my north-up terminology)
		    :param: bottom the length of the southern border
		    :param hite the lengths of the eastern and western borders
		"""
		# take stock
		nodes = [sw, se, ne, nw]
		# add constraints as needed to fully define this problem
		num_specified_nodes = sum(node is not None for node in nodes)
		if num_specified_nodes < 1:
			nodes[0] = Node(i_sw, j_sw, 0, 0)
		if num_specified_nodes < 2:
			assert nodes[1] is None, "I haven't fully generalized this line of code"
			nodes[1] = Node(i_sw, j_sw + 1, bottom, 0)

		lengths = [bottom, hite, top, hite]
		# start by replacing any unspecified Nodes with lists
		for k in range(4):
			if nodes[k] is None:
				nodes[k] = []
		# then iterate over each edge
		for k in range(4):
			a = nodes[k]
			b = nodes[(k + 1)%4]
			# if the edge is fully defined
			if type(a) is Node and type(b) is Node and a is not b:
				# use it to guess the best place for the opposite edge
				ab = np.hypot(a.x - b.x, a.y - b.y)
				scale = (ab/lengths[k])**0.5
				if type(nodes[(k + 2)%4]) is list:
					bc = lengths[(k + 1)%4]
					c = (b.x + (a.y - b.y)/ab*bc, b.y - (a.x - b.x)/ab*bc)
					nodes[(k + 2)%4].append(c)
				if type(nodes[(k + 3)%4]) is list:
					da = lengths[(k + 3)%4]
					d = (a.x + (a.y - b.y)/ab*da, a.y - (a.x - b.x)/ab*da)
					nodes[(k + 3)%4].append(d)
		# finally, average the new coordinates together to place the new Nodes
		for k in range(4):
			if type(nodes[k]) is list:
				assert len(nodes[k]) > 0, nodes
				nodes[k] = Node(i_sw + k//2, (j_sw + [0, 1, 1, 0][k])%max_j,
				                *np.mean(nodes[k], axis=0))

		self.sw: Node = nodes[0]
		self.se: Node = nodes[1]
		self.ne: Node = nodes[2]
		self.nw: Node = nodes[3]
		self.all_nodes = nodes

		self.top = top
		self.bottom = bottom
		self.hite = hite


class Section:
	def __init__(self, left_border: np.ndarray, rite_border: np.ndarray, glue_on_north: bool):
		""" a polygon that selects a portion of the globe bounded by two cuts originating
		    from the same point (the "cut_tripoint") and two continuous seams from those
		    cuts to a different point (the "glue_tripoint")
		    :param: glue_on_north whether the glue_tripoint should be the north pole (if
		                          not, it will be the south pole)
		"""
		if not (left_border[0, 0] == rite_border[0, 0] and left_border[0, 1] == rite_border[0, 1]):
			raise ValueError("the borders are supposed to start at the same point")

		self.cut_border = np.concatenate([left_border[::-1, :], rite_border])

		y_bridge = (self.cut_border[-1, 1] + self.cut_border[0, 1])/2
		if glue_on_north:
			x_pole = np.pi/2
			if self.cut_border[-1, 1] < self.cut_border[0, 1]:
				y_bridge = wrap_angle(y_bridge + np.pi)
		else:
			x_pole = -np.pi/2
			if self.cut_border[-1, 1] > self.cut_border[0, 1]:
				y_bridge = wrap_angle(y_bridge + np.pi)
		self.glue_border = np.array([self.cut_border[-1, :],
		                             [x_pole, self.cut_border[-1, 1]],
		                             [x_pole, y_bridge],
		                             [x_pole, self.cut_border[0, 1]],
		                             self.cut_border[0, :]])

		self.border = np.concatenate([self.cut_border[:-1, :], self.glue_border])


	def inside(self, x_edges: np.ndarray, y_edges: np.ndarray) -> np.ndarray:
		""" find the locus of tiles binned by x and y that are inside this section. count
		    tiles that intersect the boundary as in
		    :param: x_edges the bin edges for axis 0
		    :param: y_edges the bin edges for axis 1
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
		    :param: x_edges the bin edges for axis 0
		    :param: y_edges the bin edges for axis 1
		    :return: a boolean grid of True for shared and False for not shared
		"""
		return Section.cells_touched(x_edges, y_edges, self.glue_border)


	@staticmethod
	def cells_touched(x_edges: np.ndarray, y_edges: np.ndarray, path: np.ndarray) -> np.ndarray:
		""" find and mark each tile binned by x_edges and y_edges that is touched by this
		    polygon path
		    :param: x_edges the bin edges for axis 0
		    :param: y_edges the bin edges for axis 1
		    :param: path a n×2 array of ordered x and y coordinates
		    :return: a boolean grid of True for in and False for out
		"""
		touched = np.full((x_edges.size - 1, y_edges.size - 1), False)
		for i in range(path.shape[0] - 1):
			x0, y0 = path[i, :]
			x1, y1 = path[i + 1, :]
			i0, j0 = bin_index(x0, x_edges), bin_index(y0, y_edges)
			i1, j1 = bin_index(x1, x_edges), bin_index(y1, y_edges)
			touched[i0, j0] = True
			if i0 != i1:
				i_crossings, j_crossings = Section.grid_intersections(x_edges, y_edges[:-1], x0, y0, x1, y1, False, True)
				touched[i_crossings, j_crossings] = True
				touched[i_crossings - 1, j_crossings] = True
			if j0 != j1:
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
		    :param: x_values the values at which we should detect and bin positions (must
		                     be evenly spaced if periodic)
		    :param: y_edges the edges of the bins in which to place y values (must be
		                    evenly spaced if periodic)
		    :return: the array of x value indices and the array of y bin indices
		"""
		# make sure we don't haff to worry about periodicity issues
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
			raise ValueError("this won't work for vertical lines, I don't think.")


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
		sections.append(Section(cuts[l], cuts[(l + 1)%3], cut_tripoint[0] < 0))
	return sections


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
	nodes = np.empty((len(sections), num_ф + 1, num_λ), dtype=Node)
	cells = np.empty((len(sections), num_ф, num_λ), dtype=Cell)

	# then bild the mesh one section at a time
	main_fig, main_ax = plt.subplots()
	for l, section in enumerate(sections):
		# determine which of these cells belong in this section
		include = section.inside(ф, λ)
		share = section.shared(ф, λ)
		here_fig, here_ax = plt.subplots()
		here_ax.pcolormesh(λ, ф, np.where(include, np.where(share, 2, 1), 0))
		here_ax.plot(section.border[:, 1], section.border[:, 0], "k")
		plt.pause(.01)

		# bring in any nodes from connected layers
		for i, j in zip(*np.nonzero(share)):
			for l2 in range(cells.shape[0]): # this method of scanning l2 is kind of janky, and mite fail for more complicated section topologies
				if cells[l2, i, j] is not None:
					for node in cells[l2, i, j].all_nodes:
						nodes[l, node.i, node.j] = node

		# start at an arbitrary point
		cue = queue.Queue()
		cue.put((bin_index(section.border[0, 0], ф),
		         bin_index(section.border[0, 1], λ)))
		# then bild out from there
		while not cue.empty():
			# take the next cell that needs to be added
			i, j = cue.get()
			if include[i, j] and cells[l, i, j] is None:
				# supply it with the nodes we know and let it fill out the rest
				cells[l, i, j] = Cell(i, j, num_λ,
				                   nodes[l, i, j], nodes[l, i, (j + 1)%num_λ],
				                   nodes[l, i + 1, (j + 1)%num_λ], nodes[l, i + 1, j],
				                   EARTH_RADIUS*dλ * np.cos(ф[i + 1]), # TODO: account for eccentricity
				                   EARTH_RADIUS*dλ * np.cos(ф[i]),
				                   EARTH_RADIUS*dф)

				# update the node grid
				for node in cells[l, i, j].all_nodes:
					nodes[l, node.i, node.j] = node
				# and enforce the identities of the poles
				for k in range(1, num_λ):
					if nodes[l, 0, k] is not None:
						nodes[l, 0, :] = nodes[l, 0, k]
						break
				for k in range(1, num_λ):
					if nodes[l, num_ф, k] is not None:
						nodes[l, num_ф, :] = nodes[l, num_ф, k]
						break

				# then check each of its neibors in a breadth-first manner
				for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
					if i + di >= 0 and i + di < num_ф:
						if cells[l, i + di, (j + dj + num_λ)%num_λ] is None:
							cue.put((i + di, (j + dj + num_λ)%num_λ))

		points = np.full((nodes.shape[1], nodes.shape[2], 2), np.nan)
		for i in range(nodes.shape[1]):
			for j in range(nodes.shape[2]):
				if nodes[l, i, j] is not None:
					points[i, j, :] = [nodes[l, i, j].x, nodes[l, i, j].y]

		main_ax.plot(points[:, :, 0], points[:, :, 1], f"C{l}")
		main_ax.plot(points[:, :, 0].T, points[:, :, 1].T, f"C{l}")
		main_ax.axis("equal")
		plt.pause(.01)

	plt.show()
