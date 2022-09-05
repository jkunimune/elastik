#!/usr/bin/env python
"""
create_map_projection.py

take a basic mesh and optimize it according to a particular cost function in order to
create a new Elastic Projection.
"""
import traceback
from math import inf, pi, log2, nan

import h5py
import numpy as np
import shapefile
import tifffile
from matplotlib import pyplot as plt
from scipy.interpolate import RegularGridInterpolator

from cmap import CUSTOM_CMAP
from elastik import gradient, smooth_interpolate
from optimize import minimize
from sparse import DenseSparseArray
from util import dilate, EARTH, index_grid, Scalar, inside_region, inside_polygon, interp, \
	simplify_path, refine_path, decimate_path


# some useful custom h5 datatypes
h5_xy_tuple = [("x", float), ("y", float)]
h5_фλ_tuple = [("latitude", float), ("longitude", float)]


def get_bounding_box(points: np.ndarray) -> np.ndarray:
	""" compute the maximum and minimums of this set of points and package it as
	    [[left, bottom], [right, top]]
	"""
	return np.array([
		[np.nanmin(points[..., 0]), np.nanmin(points[..., 1])],
		[np.nanmax(points[..., 0]), np.nanmax(points[..., 1])],
	])


def downsample(full: np.ndarray, shape: tuple):
	""" decrease the size of a numpy array by setting each pixel to the mean of the pixels
	    in the original image for which it was the nearest neibor
	"""
	if full.shape == ():
		return np.full(shape, full)
	assert len(shape) == len(full.shape)
	for i in range(len(shape)):
		assert shape[i] < full.shape[i]
	reduc = np.empty(shape)
	i_reduc = (np.arange(full.shape[0])/full.shape[0]*reduc.shape[0]).astype(int)
	j_reduc = (np.arange(full.shape[1])/full.shape[1]*reduc.shape[1]).astype(int)
	for i in range(shape[0]):
		for j in range(shape[1]):
			reduc[i][j] = np.mean(full[i_reduc == i][:, j_reduc == j])
	return reduc


def follow_graph(progression: np.ndarray, frum: np.ndarray, until: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
	""" take with a markov-chain-like-thing in the form of an array of indices and follow
	    it to some conclusion.
	    :param progression: the indices of the nodes that follow from each node
	    :param frum: the starting state, an array of indices
	    :param until: a boolean array indicating points that don't need to follow to
	                        the next part of the graph
	"""
	state = frum.copy()
	distance_traveld = np.zeros(state.shape)
	arrived = until[state]
	while np.any(~arrived):
		if np.any((~arrived) & (progression[state] == state)):
			raise ValueError("this importance graff was about to cause an infinite loop.")
		state[~arrived] = progression[state[~arrived]]
		distance_traveld[~arrived] += 1
		arrived = until[state]
	return state, distance_traveld


def get_interpolation_weits(distance_a, distance_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
	""" compute the weits needed to linearly interpolate a point between two fixed points,
	    given the distance of each respective reference to the point of interpolation.
	"""
	weits_a = np.empty(distance_a.shape)
	normal = distance_a + distance_b != 0
	weits_a[normal] = distance_b[normal]/(distance_a + distance_b)[normal]
	weits_a[~normal] = 1
	weits_b = 1 - weits_a
	return weits_a, weits_b


def enumerate_nodes(mesh: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
	""" take an array of positions for a mesh and generate the list of unique nodes,
	    returning mappings from the old mesh to the new indices and the n×2 list positions
	"""
	# first flatten and sort the positions
	node_positions = mesh.reshape((-1, mesh.shape[-1]))
	node_positions, node_indices = np.unique(node_positions, axis=0, return_inverse=True)
	node_indices = node_indices.reshape(mesh.shape[:-1])

	# then remove any nans, which in fact represent the absence of a node
	nan_index = np.nonzero(np.isnan(node_positions[:, 0]))[0][0]
	node_positions = node_positions[:nan_index, :]
	node_indices[node_indices >= nan_index] = -1

	return node_indices, node_positions


def enumerate_cells(node_indices: np.ndarray, values: list[np.ndarray], scales: list[np.ndarray],
                    dΦ: np.ndarray, dΛ: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	""" take an array of nodes and generate the list of cells in which the elastic energy
	    should be calculated.
	    :param node_indices: the lookup table that tells you the index in the position
	                         vector at which is stored each node at each location in the
	                         mesh
	    :param values: the relative importance of each cell in the cell matrix
	    :param scales: the desired relative areal scale factor for each location in the cell matrix
	    :param dΦ: the spacing between each adjacent row of nodes (km)
	    :param dΛ: the spacing between adjacent nodes in each row (km)
	    :return: cell_definitions: the list of cells, each defined by a set of seven
	                               indices (the section index, the two indices specifying
	                               its location the matrix, and the indices of the four
	                               vertex nodes (two of them are probably the same node)
	                               in the node vector
	             cell_weights: the volume of each cell for elastic-energy-summing porpoises
	             cell_scales: the desired relative linear scale factor for each cell

	"""
	# start off by resampling these in a useful way
	for h in range(node_indices.shape[0]):
		values[h] = downsample(values[h], node_indices.shape[1:])
		scales[h] = downsample(scales[h], node_indices.shape[1:])
	values = np.stack(values) # TODO: make values on glue borders a little smarter; shared cells should share weights
	scales = np.stack(scales)

	# assemble a list of all possible cells
	h, i, j = index_grid((node_indices.shape[0],
	                      node_indices.shape[1] - 1,
	                      node_indices.shape[2] - 1))
	h, i, j = h.ravel(), i.ravel(), j.ravel()
	cell_definitions = np.empty((0, 12), dtype=int)
	for di in range(0, 2):
		for dj in range(0, 2):
			# define them by their indices and neiboring node indices
			west_node = node_indices[h, i + di, j]
			east_node = node_indices[h, i + di, j + 1]
			south_node = node_indices[h, i,     j + dj]
			north_node = node_indices[h, i + 1, j + dj]
			cell_definitions = np.concatenate([
				cell_definitions,
				np.stack([h, i, j, i + di, i + 1 - di, # these first five will get chopd off once I'm done with them
					      h, i + di, j + dj, # these middle three are for generic spacially dependent stuff
				          west_node, east_node, # these bottom four are the really important indices
				          south_node, north_node], axis=-1)])

	# then remove all duplicates
	_, unique_indices = np.unique(cell_definitions[:, -4:], axis=0, return_index=True)
	cell_definitions = cell_definitions[unique_indices, :]

	# and remove the ones that rely on missingnodes or that rely on the poles too many times
	missing_node = np.any(cell_definitions[:, -4:] == -1, axis=1)
	degenerate = cell_definitions[:, -4] == cell_definitions[:, -3]
	cell_definitions = cell_definitions[~(missing_node | degenerate), :]

	# you can pull apart the cell definitions now
	cell_hs = cell_definitions[:, 0]
	cell_is = cell_definitions[:, 1]
	cell_js = cell_definitions[:, 2]
	cell_node1_is = cell_definitions[:, 3]
	cell_node2_is = cell_definitions[:, 4]
	cell_definitions = cell_definitions[:, 5:]

	# finally, calculate their areas and stuff
	A_1 = dΦ[cell_node1_is]*dΛ[cell_node1_is]
	A_2 = dΦ[cell_node2_is]*dΛ[cell_node2_is]
	cell_areas = (3*A_1 + A_2)/16/(4*np.pi*EARTH.R**2)

	cell_weights = cell_areas*values[cell_hs, cell_is, cell_js]
	cell_scales = np.sqrt(scales[cell_hs, cell_is, cell_js]) # this sqrt converts it from areal scale to linear

	return cell_definitions, cell_weights, cell_scales


def mesh_skeleton(lookup_table: np.ndarray, factor: int, ф: np.ndarray
                  ) -> tuple[np.ndarray | DenseSparseArray | Scalar, np.ndarray | DenseSparseArray | Scalar]:
	""" create a pair of inverse functions that transform points between the full space of
	    possible meshes and a reduced space with fewer degrees of freedom. the idea here
	    is to identify 80% or so of the nodes that can form a skeleton, and from which the
	    remaining nodes can be interpolated. to that end, each parallel will have some
	    number of regularly spaced key nodes, and all nodes adjacent to an edge will
	    be keys, as well, and all nodes that aren't key nodes will be ignored when
	    reducing the position vector and interpolated from neibors when restoring it.
	    :param lookup_table: the index of each node's position in the state vector
	    :param factor: approximately how much the resolution should decrease
	    :param ф: the latitudes of the nodes
	    :return: a matrix that linearly reduces a set of node positions to just the bare
	             skeleton, and a matrix that linearly reconstructs the missing node
	             positions from a reduced set.  don't worry about their exact types;
	             they'll both support matrix multiplication with '@'.
	"""
	n_full = np.max(lookup_table) + 1
	# start by filling out these connection graffs, which are nontrivial because of the layers
	east_neibor = np.full(n_full, -1)
	west_neibor = np.full(n_full, -1)
	north_neibor = np.full(n_full, -1)
	south_neibor = np.full(n_full, -1)
	for h in range(lookup_table.shape[0]):
		for i in range(lookup_table.shape[1]):
			for j in range(lookup_table.shape[2]):
				if lookup_table[h, i, j] != -1:
					if j - 1 >= 0 and lookup_table[h, i, j - 1] != -1:
						west_neibor[lookup_table[h, i, j]] = lookup_table[h, i, j - 1]
					if j + 1 < lookup_table.shape[2] and lookup_table[h, i, j + 1] != -1:
						east_neibor[lookup_table[h, i, j]] = lookup_table[h, i, j + 1]
					if i - 1 >= 0 and lookup_table[h, i - 1, j] != -1:
						south_neibor[lookup_table[h, i, j]] = lookup_table[h, i - 1, j]
					if i + 1 < lookup_table.shape[1] and lookup_table[h, i + 1, j] != -1:
						north_neibor[lookup_table[h, i, j]] = lookup_table[h, i + 1, j]

	# then decide which nodes should be independently defined in the skeleton
	has_defined_neibors = np.full(n_full + 1, False) # (this array has an extra False at the end so that -1 works nicely)
	is_defined = np.full(n_full, False)
	# start by marking some evenly spaced interior points
	for h in range(lookup_table.shape[0]):
		for i in range(lookup_table.shape[1]):
			east_west_factor = int(round(factor/np.cos(ф[i])))
			for j in range(lookup_table.shape[2]):
				if lookup_table[h, i, j] != -1:
					important_row = (min(i, ф.size - 1 - i)%factor == 0)
					important_col = (j%east_west_factor == 0)
					has_defined_neibors[lookup_table[h, i, j]] |= important_row
					is_defined[lookup_table[h, i, j]] |= important_col
	# then make sure we define enuff points at each edge to keep it all fully defined
	has_defined_neibors[:-1] |= (north_neibor == -1) | (south_neibor == -1)
	is_defined |= (~has_defined_neibors[east_neibor]) | (~has_defined_neibors[west_neibor])
	is_defined &= has_defined_neibors[:-1]

	reindex = np.where(is_defined, np.cumsum(is_defined) - 1, -1)
	n_partial = np.max(reindex) + 1

	# then decide how to define the ones that aren't defined
	n_reference, n_distance = follow_graph(north_neibor, frum=np.arange(n_full), until=has_defined_neibors)
	ne_reference, ne_distance = follow_graph(east_neibor, frum=n_reference, until=is_defined)
	nw_reference, nw_distance = follow_graph(west_neibor, frum=n_reference, until=is_defined)
	s_reference, s_distance = follow_graph(south_neibor, frum=np.arange(n_full), until=has_defined_neibors)
	se_reference, se_distance = follow_graph(east_neibor, frum=s_reference, until=is_defined)
	sw_reference, sw_distance = follow_graph(west_neibor, frum=s_reference, until=is_defined)
	n_weit, s_weit = get_interpolation_weits(n_distance, s_distance)
	ne_weit, nw_weit = get_interpolation_weits(ne_distance, nw_distance)
	se_weit, sw_weit = get_interpolation_weits(se_distance, sw_distance)
	defining_indices = np.stack([
		ne_reference, nw_reference, se_reference, sw_reference,
	], axis=1)
	defining_weits = np.stack([
		n_weit * ne_weit,
		n_weit * nw_weit,
		s_weit * se_weit,
		s_weit * sw_weit,
	], axis=1)

	# put the conversions together and return them as functions
	reduction = DenseSparseArray.identity(n_full)[is_defined, :]
	restoration = DenseSparseArray.from_coordinates(
		[n_partial], np.expand_dims(reindex[defining_indices], axis=-1), defining_weits)
	return reduction, restoration


def compute_principal_strains(positions: np.ndarray,
                              cell_definitions: np.ndarray, cell_scales: np.ndarray,
                              dΦ: np.ndarray, dΛ: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
	""" take a set of cell definitions and 2D coordinates for each node, and calculate
	    the Tissot-ellipse semiaxes of each cell.
	    :param positions: the vector specifying the location of each node in the map plane
	    :param cell_definitions: the list of cells, each defined by seven indices
	    :param cell_scales: the linear scale factor for each cell
	    :param dΦ: the distance between adjacent rows of nodes (km)
	    :param dΛ: the distance between adjacent nodes in each row (km)
	"""
	i = cell_definitions[:, 1]

	west = positions[cell_definitions[:, 3], :]
	east = positions[cell_definitions[:, 4], :]
	F_λ = ((east - west)/(dΛ[i]/cell_scales)[:, np.newaxis])
	dxdΛ, dydΛ = F_λ[:, 0], F_λ[:, 1]

	south = positions[cell_definitions[:, 5], :]
	north = positions[cell_definitions[:, 6], :]
	F_ф = ((north - south)/(dΦ[i]/cell_scales)[:, np.newaxis])
	dxdΦ, dydΦ = F_ф[:, 0], F_ф[:, 1]

	trace = np.sqrt((dxdΛ + dydΦ)**2 + (dxdΦ - dydΛ)**2)/2
	antitrace = np.sqrt((dxdΛ - dydΦ)**2 + (dxdΦ + dydΛ)**2)/2
	return trace + antitrace, trace - antitrace


def project(points: list[tuple[float, float]] | np.ndarray, ф_mesh: np.ndarray, λ_mesh: np.ndarray,
            nodes: np.ndarray, section_borders: list[np.ndarray] = None, section_index: int = None) -> DenseSparseArray | np.ndarray:
	""" take some points, giving all sections of the map projection, and project them into the plane, representing them
	    either with their resulting cartesian coordinates or as matrix that multiplies by the vector of node positions
	    to produce an array of points on the border that can be used to compute the bounding box and enforce constraints
	    :param points: the spherical coordinates of the points to project
	    :param ф_mesh: the latitudes at which the map position is defined
	    :param λ_mesh: the longitudes at which the map position is defined
	    :param section_borders: the list of the borders for all the sections
	    :param section_index: the index of the section to use for all points
	    :param nodes: either a) a 4d array of x and y coordinates for each section, from which the coordinates of the
	                            projected points will be determined, or
	                         b) a 3d array of node indices for each section, indicating that the result should be a
	                            matrix that you can multiply by the node positions later.
	"""
	if section_borders is None and section_index is None:
		raise ValueError("at least one of section_borders and section_index must not be none.")

	positions_known = nodes.ndim == 4

	# first, we set up an identity matrix of sorts, if the positions aren't currently known
	if positions_known:
		valid = np.isfinite(nodes[:, :, :, 0])
	else:
		valid = nodes != -1
		num_nodes = np.max(nodes) + 1
		nodes = DenseSparseArray.identity(num_nodes, add_zero=True).to_array_array()[nodes]

	# next, we must identify any nodes that exist in multiple layers
	hs = np.arange(nodes.shape[0])
	shared = np.tile(np.any(valid & np.all(np.reshape(nodes[hs, ...] == nodes[hs - 1, ...],
	                                                  newshape=nodes.shape[:3] + (-1,)),
	                                       axis=3), axis=0),
	                 (nodes.shape[0], 1, 1))

	# and start calculating gradients
	ф_gradients = np.empty(nodes.shape, dtype=nodes.dtype)
	λ_gradients = np.empty(nodes.shape, dtype=nodes.dtype)
	for mask in [valid, shared]:
		ф_gradients[mask, ...] = gradient(nodes, ф_mesh, where=mask, axis=1)[mask, ...]
		λ_gradients[mask, ...] = gradient(nodes, λ_mesh, where=mask, axis=2)[mask, ...]
	for h in range(nodes.shape[0]):
		for i in range(nodes.shape[1]):
			for j in range(nodes.shape[2]):
				if valid[h, i, j]:
					assert ф_gradients[h, i, j] is not None, (h, i, j)

	# finally, interpolate
	result = []
	for point in points:
		h = section_index
		if h is None:
			for trial_h, border in zip(hs, section_borders): # on the correct section
				if inside_region(*point, border, period=2*pi):
					h = trial_h
					break
		result.append(smooth_interpolate(point, (ф_mesh, λ_mesh), nodes[h, ...],
		                                 (ф_gradients[h, ...], λ_gradients[h, ...])))

	# convert to the correct type
	if positions_known:
		return np.array(result)
	else:
		return DenseSparseArray.concatenate(result)


def inverse_project(points: np.ndarray, ф_mesh: np.ndarray, λ_mesh: np.ndarray,
                    nodes: np.ndarray, section_borders: list[np.ndarray] = None, section_index: int = None) -> DenseSparseArray | np.ndarray:
	""" take some points, specifying a section of the map projection, and project them from the plane back to the globe,
	    representing the result as the resulting latitudes and longitudes
	    :param ф_mesh: the latitudes at which the map position is defined
	    :param λ_mesh: the longitudes at which the map position is defined
	    :param section_borders: the boundary of each section in spherical coordinates
	    :param section_index: the index of the section to which we should project everything
	    :param nodes: the positions of the nodes
	    :param points: the spherical coordinates of the points to project
	"""
	if section_borders is None and section_index is None:
		raise ValueError("at least one of section_borders and section_index must not be none.")

	if section_index is not None:
		hs = [section_index]
	else:
		hs = range(nodes.shape[0])

	# do each point one at a time, since this doesn't need to be super fast
	sectioned_results = np.full((len(hs),) + points.shape, nan)
	for point_index, point in enumerate(points.reshape((-1, 2))):
		point_index = np.unravel_index(point_index, points.shape[:-1])
		print(f"{point_index}/{points.shape[:-1]}")
		for h in hs:
			for i in range(1, nodes.shape[1]):
				for j in range(1, nodes.shape[2]):
					# look for a cell that contains it
					if np.all(np.isfinite(nodes[i-1:i+1, j-1:j+1])):
						sw = nodes[h, i - 1, j - 1]
						se = nodes[h, i - 1, j]
						nw = nodes[h, i, j - 1]
						ne = nodes[h, i, j]
						if inside_polygon(*point, np.array([ne, nw, sw, se]), convex=True):
							# do inverse 2d linear interpolation (it's harder than one mite expect!)
							coords = [nan, nan]
							things = [(0, ф_mesh, i, [[sw, se], [nw, ne]]), (1, λ_mesh, j, [[sw, nw], [se, ne]])]
							for f, axis, index, corners in things:
								for g in [0, 1]:
									x, y = point[g], point[1 - g]
									y0 = interp(x,
									            corners[0][0][    g], corners[0][1][    g],
									            corners[0][0][1 - g], corners[0][1][1 - g])
									y1 = interp(x,
									            corners[1][0][    g], corners[1][1][    g],
									            corners[1][0][1 - g], corners[1][1][1 - g])
									if (y0 < y) != (y1 < y):
										coords[f] = interp(y, y0, y1, axis[index - 1], axis[index])
										break
							sectioned_results[(h, *point_index, slice(None))] = coords
							break

	# deal with multiple possible projections for each point, if there were multiple sections given
	if section_borders is not None:
		result = np.full(points.shape, nan)
		for h in hs:
			inside_h = inside_region(sectioned_results[h, ..., 0],
			                         sectioned_results[h, ..., 1],
			                         section_borders[h], period=2*pi)
			result[inside_h, :] = sectioned_results[h, inside_h, :]
	else:
		result = sectioned_results[0, ...]

	return result


def load_options(filename: str) -> dict[str, str]:
	""" load a simple colon-separated text file """
	options = dict()
	with open(f"../spec/options_{filename}.txt", "r", encoding="utf-8") as file:
		for line in file.readlines():
			key, value = line.split(":")
			options[key.strip()] = value.strip()
	return options


def load_pixel_values(filename: str, cut_set: str, num_sections: int, minimum=-inf) -> list[np.ndarray]:
	""" load and resample a generic 2D raster image """
	if filename == "uniform":
		return [np.array(1.)]*num_sections
	else:
		values = []
		for h in range(num_sections):
			values.append(np.maximum(
				minimum, tifffile.imread(f"../spec/pixels_{cut_set}_{h}_{filename}.tif")))
		return values


def load_coastline_data(reduction=2) -> list[np.ndarray]:
	coastlines = []
	with shapefile.Reader(f"../data/ne_110m_coastline.zip") as shapef:
		for shape in shapef.shapes():
			if len(shape.points) > 3*reduction:
				coastlines.append(np.radians(shape.points)[::reduction, ::-1])
	return coastlines


def load_mesh(filename: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:
	""" load the ф values, λ values, node locations, and section borders from a HDF5
	    file, in that order.
	"""
	with h5py.File(f"../spec/mesh_{filename}.h5", "r") as file:
		ф = np.radians(file["section0/latitude"])
		λ = np.radians(file["section0/longitude"])
		num_sections = file.attrs["num_sections"]
		mesh = np.empty((num_sections, ф.size, λ.size, 2))
		sections = []
		for h in range(num_sections):
			mesh[h, :, :, :] = file[f"section{h}/projection"]
			sections.append(np.radians(file[f"section{h}/border"][:, :]))
	return ф, λ, mesh, sections


def show_mesh(fit_positions: np.ndarray, all_positions: np.ndarray, velocity: np.ndarray,
              values: list[float], grads: list[float], final: bool,
              ф_mesh: np.ndarray, λ_mesh: np.ndarray, dΦ: np.ndarray, dΛ: np.ndarray,
              mesh_index: np.ndarray, cell_definitions: np.ndarray,
              cell_weights: np.ndarray, cell_scales: np.ndarray,
              coastlines: list[np.array], border: np.ndarray | DenseSparseArray,
              map_axes: plt.Axes, hist_axes: plt.Axes,
              valu_axes: plt.Axes, diff_axes: plt.Axes) -> None:
	map_axes.clear()
	mesh = np.where(mesh_index[:, :, :, np.newaxis] != -1,
	                all_positions[mesh_index, :], nan)
	for h in range(mesh.shape[0]):
		# plot the underlying mesh for each section
		map_axes.plot(mesh[h, :, :, 0], mesh[h, :, :, 1], f"#bbb", linewidth=.3) # TODO: zoom in and stuff?
		map_axes.plot(mesh[h, :, :, 0].T, mesh[h, :, :, 1].T, f"#bbb", linewidth=.3)
		# crudely project and plot the coastlines onto each section
		project = RegularGridInterpolator([ф_mesh, λ_mesh], mesh[h, :, :, :],
		                                  bounds_error=False, fill_value=nan)
		for line in coastlines:
			projected_line = project(line)
			map_axes.plot(projected_line[:, 0], projected_line[:, 1], f"#000", linewidth=.8, zorder=2)
		# plot the outline of the mesh
		border_points = border @ all_positions
		loop = np.arange(-1, border_points.shape[0])
		map_axes.plot(border_points[loop, 0], border_points[loop, 1], f"#000", linewidth=1.3, zorder=2)
	if not final:
		# indicate the speed of each node
		map_axes.scatter(fit_positions[:, 0], fit_positions[:, 1], s=2,
		                 c=-np.linalg.norm(velocity, axis=1),
		                 cmap=CUSTOM_CMAP["speed"]) # TODO: zoom in and rotate automatically

	a, b = compute_principal_strains(all_positions, cell_definitions, cell_scales, dΦ, dΛ)

	# mark any nodes with nonpositive principal strains
	worst_cells = np.nonzero((a <= 0) | (b <= 0))[0]
	for cell in worst_cells:
		h, i, j, east, west, north, south = cell_definitions[cell, :]
		plt.plot(all_positions[[east, west], 0], all_positions[[east, west], 1], "#f60", linewidth=1.3)
		plt.plot(all_positions[[north, south], 0], all_positions[[north, south], 1], "#f60", linewidth=1.3)
	map_axes.axis("equal")

	# histogram the principal strains
	hist_axes.clear()
	hist_axes.hist2d(np.concatenate([a, b]),
	                 np.concatenate([b, a]),
	                 weights=np.tile(cell_weights, 2),
	                 bins=np.linspace(0, 2, 41),
	                 cmap=CUSTOM_CMAP["density"])
	hist_axes.axis("square")

	# plot the error function over time
	valu_axes.clear()
	valu_axes.plot(values)
	valu_axes.set_xlim(len(values) - 1000, len(values))
	valu_axes.set_ylim(0, 3*values[-1])
	valu_axes.minorticks_on()
	valu_axes.yaxis.set_tick_params(which='both')
	valu_axes.grid(which="both", axis="y")

	# plot the convergence criteria over time
	diff_axes.clear()
	diffs = -np.diff(values)/values[1:]
	if diffs.size > 0:
		diff_axes.scatter(np.arange(1, len(values)), diffs, s=1, zorder=10)
	diff_axes.scatter(np.arange(len(grads)), grads, s=1, zorder=10)
	ylim = max(2e-2, diffs.min(where=diffs != 0, initial=np.min(grads))*1e3)
	diff_axes.set_ylim(ylim/1e3, ylim)
	diff_axes.set_yscale("log")
	diff_axes.grid(which="major", axis="y")


def save_mesh(name: str, descript: str, ф: np.ndarray, λ: np.ndarray, mesh: np.ndarray,
              section_borders: list[np.ndarray], section_names: list[str], projected_border: np.ndarray) -> None:
	""" save all of the important map projection information as a HDF5 and text file.
	    :param name: the name of this elastick map projection
	    :param descript: a short description of the map projection, to be included in the
	                     HDF5 file as an attribute.
	    :param ф: the m latitude values at which the projection is defined
	    :param λ: the l longitude values at which the projection is defined
	    :param mesh: an n×m×l×2 array of the x and y coordinates at each point
	    :param section_borders: a list of the borders of the n sections. each one is an
	                            o×2 array of x and y coordinates, starting and ending at
	                            the same point.
	    :param section_names: a list of the names of the n sections. these will be added
	                          to the HDF5 file as attributes.
	    :param projected_border:
	"""
	assert len(section_borders) == len(section_names)

	# start by calculating some things
	projected_border = decimate_path(projected_border, resolution=10)
	((left, bottom), (right, top)) = get_bounding_box(projected_border)

	# do the self-explanatory HDF5 file
	with h5py.File(f"../projection/elastik-{name}.h5", "w") as file:
		file.attrs["name"] = name
		file.attrs["description"] = descript
		file.attrs["num_sections"] = len(section_borders)
		file.create_dataset("projected_border", shape=(projected_border.shape[0],), dtype=h5_xy_tuple)
		file["projected_border"]["x"] = projected_border[:, 0]
		file["projected_border"]["y"] = projected_border[:, 1]
		file.create_dataset("bounding_box", shape=(2,), dtype=h5_xy_tuple)
		file["bounding_box"]["x"] = [left, right]
		file["bounding_box"]["y"] = [bottom, top]
		file["bounding_box"].attrs["units"] = "km"
		file["sections"] = [f"section{i}" for i in range(len(section_borders))]

		for h in range(len(section_borders)):
			i_relevant = dilate(np.any(~np.isnan(mesh[h, :, :, 0]), axis=1), 1)
			num_ф = np.sum(i_relevant)
			j_relevant = dilate(np.any(~np.isnan(mesh[h, 1:-1, :, 0]), axis=0), 1)
			num_λ = np.sum(j_relevant)

			group = file.create_group(f"section{h}")
			group.attrs["name"] = section_names[h]
			group["latitude"] = np.degrees(ф[i_relevant]) # TODO: internationalize
			group["latitude"].attrs["units"] = "°"
			group["latitude"].make_scale()
			group["longitude"] = np.degrees(λ[j_relevant])
			group["longitude"].make_scale()
			group["longitude"].attrs["units"] = "°"
			group.create_dataset("projection", shape=(num_ф, num_λ), dtype=h5_xy_tuple)
			group["projection"]["x"] = mesh[h, i_relevant][:, j_relevant, 0]
			group["projection"]["y"] = mesh[h, i_relevant][:, j_relevant, 1]
			group["projection"].attrs["units"] = "km"
			group["projection"].dims[0].attach_scale(group["latitude"])
			group["projection"].dims[1].attach_scale(group["longitude"])
			group.create_dataset("border", shape=(section_borders[h].shape[0],), dtype=h5_фλ_tuple)
			group["border"]["latitude"] = np.degrees(section_borders[h][:, 0])
			group["border"]["longitude"] = np.degrees(section_borders[h][:, 1])
			group["border"].attrs["units"] = "°"
			((left_h, bottom_h), (right_h, top_h)) = get_bounding_box(mesh[h, :, :, :])
			group.create_dataset("bounding_box", shape=(2,), dtype=h5_xy_tuple)
			group["bounding_box"]["x"] = [max(left, left_h), min(right, right_h)]
			group["bounding_box"]["y"] = [max(bottom, bottom_h), min(top, top_h)]
			group["bounding_box"].attrs["units"] = "km"

	# then save a simpler but larger and less explanatory txt file
	raster_resolution = mesh.shape[1]
	x_raster = np.linspace(left, right, raster_resolution)
	y_raster = np.linspace(bottom, top, raster_resolution)
	projected_raster = np.degrees(inverse_project(
		np.transpose(np.meshgrid(x_raster, y_raster, indexing="xy"), (1, 2, 0)),
		ф, λ, mesh, section_borders=section_borders)) # TODO: this needs to be a *lot* faster

	with open(f"../projection/elastik-{name}.csv", "w") as f:
		f.write(f"elastik {name} projection ({mesh.shape[0]} sections):\n") # the number of sections
		for h in range(mesh.shape[0]):
			f.write(f"border ({section_borders[h].shape[0]} vertices):\n") # the number of section border vertices
			for i in range(section_borders[h].shape[0]):
				f.write(f"{section_borders[h][i, 0]:.6f},{section_borders[h][i, 1]:.6f}\n") # the section border vertices (°)
			f.write(f"projection ({mesh.shape[1]}x{mesh.shape[2]} points):\n") # the shape of the section mesh
			for i in range(mesh.shape[1]):
				for j in range(mesh.shape[2]):
					f.write(f"{mesh[h, i, j, 0]:.3f},{mesh[h, i, j, 1]:.3f}") # the section mesh points (km)
					if j != mesh.shape[2] - 1:
						f.write(", ")
				f.write("\n")
		f.write(f"projected border ({projected_border.shape[0]} vertices):\n") # the number of map edge vertices
		for i in range(projected_border.shape[0]):
			f.write(f"{projected_border[i, 0]:.3f},{projected_border[i, 1]:.3f}\n") # the map edge vertices (km)
		f.write(f"projected raster ({projected_raster.shape[0]}x{projected_raster.shape[1]}):\n") # the shape of the sample raster
		f.write(f"{left}-{right}, {bottom}-{top}\n") # the bounding box of the sample raster
		for i in range(projected_raster.shape[0]):
			for j in range(projected_raster.shape[1]):
				f.write(f"{projected_raster[i, j, 0]:.6f},{projected_raster[i, j, 1]:.6f}") # the sample raster (°)
				if j != projected_raster.shape[1] - 1:
					f.write(", ")
			f.write("\n")


def create_map_projection(configuration_file: str):
	""" create a map projection
	    :param configuration_file: "oceans" | "continents" | "countries"
	"""
	configure = load_options(configuration_file)
	print(f"loaded options from {configuration_file}")
	ф_mesh, λ_mesh, mesh, section_borders = load_mesh(configure["cuts"])
	print(f"loaded a {np.sum(np.isfinite(mesh[:, :, :, 0]))}-node mesh")
	scale = load_pixel_values(configure["scale"], configure["cuts"], mesh.shape[0], .03)
	print(f"loaded the {configure['scale']} map as the scale")
	weights = load_pixel_values(configure["weights"], configure["cuts"], mesh.shape[0], .03)
	print(f"loaded the {configure['weights']} map as the weights")
	width, height = (float(value) for value in configure["size"].split(","))
	print(f"setting the maximum map size to {width}×{height} km")

	# assume the coordinates are more or less evenly spaced
	dΦ = EARTH.a*(1 - EARTH.e2)*(1 - EARTH.e2*np.sin(ф_mesh)**2)**(3/2)*(ф_mesh[1] - ф_mesh[0])
	dΛ = EARTH.a*(1 + (1 - EARTH.e2)*np.tan(ф_mesh)**2)**(-1/2)*(λ_mesh[1] - λ_mesh[0])

	# reformat the nodes into a list without gaps or duplicates
	node_indices, node_positions = enumerate_nodes(mesh)

	# and then do the same thing for cell corners
	cell_definitions, cell_weights, cell_scales = enumerate_cells(node_indices, weights, scale, dΦ, dΛ)

	# define functions that can define the node positions from a reduced set of them
	transformations = []
	progression = np.ceil(np.geomspace(ф_mesh.size/10, 1.,
	                                   int(log2(ф_mesh.size/10)) + 1))
	for factor in progression:
		transformations.append(mesh_skeleton(node_indices, factor, ф_mesh))
	transformations.append((Scalar(1), Scalar(1))) # finishing with the full unreduced set

	# set up the fitting constraints that will force the map to fit inside a box
	border_points = []
	for h, border in enumerate(section_borders):
		first_pole, last_pole = np.nonzero(abs(border[:, 0]) == pi/2)[0][[0, -1]]
		border = np.concatenate([border[last_pole:], border[1:first_pole + 1]])
		border_points.append(project(refine_path(border, pi/100, period=2*pi),
		                             ф_mesh, λ_mesh, node_indices, section_index=h))
	border_points = simplify_path(DenseSparseArray.concatenate(border_points), cyclic=True)
	map_size = ([-width/2, -height/2], [width/2, height/2])

	# load the coastline data from Natural Earth
	coastlines = load_coastline_data()

	# set up the plotting axes
	small_fig = plt.figure(figsize=(3, 5), num=f"Elastik-{configuration_file} fitting")
	gridspecs = (plt.GridSpec(3, 1, height_ratios=[2, 1, 1]),
	             plt.GridSpec(3, 1, height_ratios=[2, 1, 1], hspace=0))
	hist_axes = small_fig.add_subplot(gridspecs[0][0, :])
	valu_axes = small_fig.add_subplot(gridspecs[1][1, :])
	diff_axes = small_fig.add_subplot(gridspecs[1][2, :], sharex=valu_axes)
	main_fig, map_axes = plt.subplots(figsize=(7, 5), num=f"Elastik-{configuration_file}")

	values, grads = [], []

	# define the objective functions
	def compute_energy_aggressive(positions: np.ndarray) -> float: # one that aggressively pushes the mesh to have all positive strains
		a, b = compute_principal_strains(restore @ positions,
		                                 cell_definitions, cell_scales, dΦ, dΛ)
		if np.all(a > 0) and np.all(b > 0):
			return -inf
		elif np.any(a < -100) or np.any(b < -100): # make this check to avoid annoying overflow warnings
			return inf
		else:
			a_term = np.exp(-6*a)
			b_term = np.exp(-6*b)
			return (a_term + b_term).sum()

	def compute_energy_lenient(positions: np.ndarray) -> float: # one that works in all domains
		a, b = compute_principal_strains(restore @ positions,
		                                 cell_definitions, cell_scales, dΦ, dΛ)
		if np.all(a > 0) and np.all(b > 0):
			return -inf
		else:
			scale_term = (a + b - 2)**2
			shape_term = (a - b)**2
			return ((scale_term + 2*shape_term)*cell_weights).sum()

	def compute_energy_strict(positions: np.ndarray) -> float: # and one that only works when all strains are positive
		a, b = compute_principal_strains(restore @ positions,
		                                 cell_definitions, cell_scales, dΦ, dΛ)
		if np.any(a <= 0) or np.any(b <= 0):
			return inf
		else:
			ab = a*b
			scale_term = (ab**2 - 1)/2 - np.log(ab)
			shape_term = (a - b)**2
			return ((scale_term + 2*shape_term)*cell_weights).sum()

	def plot_status(positions: np.ndarray, value: float, grad: np.ndarray, step: np.ndarray, final: bool) -> None:
		values.append(value)
		grads.append(np.linalg.norm(grad)*EARTH.R)
		if len(values) == 1 or np.random.random() < 1e-1 or final:
			show_mesh(positions, restore @ positions, step, values, grads, final,
			          ф_mesh, λ_mesh, dΦ, dΛ, node_indices,
			          cell_definitions, cell_weights, cell_scales,
			          coastlines, border_points,
			          map_axes, hist_axes, valu_axes, diff_axes)
			main_fig.canvas.draw()
			small_fig.canvas.draw()
			plt.pause(.01)

	# then minimize!
	print("begin fitting process.")
	try:
		# progress from the coarsest transformd mesh to finer and finer ones
		for reduce, restore in transformations:
			node_positions = reduce @ node_positions
			node_positions = minimize(func=compute_energy_strict,
			                          backup_func=compute_energy_lenient,
			                          guess=node_positions,
			                          bounds=[(border_points @ restore, map_size[0], map_size[1])],
			                          report=plot_status,
			                          tolerance=1e-3/EARTH.R)
			node_positions = restore @ node_positions

		# the mesh should, at some point, become well-behaved
		if not np.all(np.greater(compute_principal_strains(
			node_positions, cell_definitions, cell_scales, dΦ, dΛ), 0)):
			# if it doesn't do a final pass using the aggressive objective function to whip it into shape
			node_positions = minimize(func=compute_energy_strict,
			                          backup_func=compute_energy_aggressive,
			                          guess=node_positions,
			                          bounds=[(border_points, map_size[0], map_size[1])],
			                          report=plot_status,
			                          tolerance=1e-3/EARTH.R)

	except RuntimeError as e:
		traceback.print_exc()
		small_fig.canvas.manager.set_window_title("Error!")
		plt.show()
		raise e

	print("end fitting process.")

	# apply the optimized vector back to the mesh
	mesh = node_positions[node_indices, :]
	mesh[node_indices == -1, :] = nan

	# and save it!
	save_mesh(configure["name"], configure["descript"],
	          ф_mesh, λ_mesh, mesh,
	          section_borders, configure["section_names"].split(","),
	          border_points @ node_positions)

	print(f"elastik {configure['name']} projection saved!")


if __name__ == "__main__":
	# create_map_projection("oceans")
	create_map_projection("continents")
	# create_map_projection("countries")

	plt.show()
