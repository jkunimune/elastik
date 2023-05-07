#!/usr/bin/env python
"""
create_map_projection.py

take a basic mesh and optimize it according to a particular cost function in order to
create a new Elastic Projection.
"""
from __future__ import annotations

import json
import logging
import sys
import threading
from math import inf, pi, log2, nan, floor, isfinite
from typing import Iterable, Sequence

import h5py
import numpy as np
import shapefile
import tifffile
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from scipy.interpolate import RegularGridInterpolator

from cmap import CUSTOM_CMAP
from optimize import minimize_with_bounds
from sparse import SparseNDArray
from util import dilate, EARTH, index_grid, Scalar, inside_region, interp, \
	simplify_path, refine_path, decimate_path, rotate_and_shift, fit_in_rectangle, Tensor, bin_index, \
	inside_polygon, search_out_from

logging.basicConfig(
	level=logging.INFO,
	format="%(asctime)s | %(levelname)s | %(message)s",
	datefmt="%b %d %H:%M",
	handlers=[
		logging.FileHandler("../projection/output.log"),
		logging.StreamHandler(sys.stdout)
	]
)


MIN_WEIGHT = .03 # the ratio of the whitespace weight to the subject weight


# some useful custom h5 datatypes
h5_xy_tuple = [("x", float), ("y", float)]
h5_фλ_tuple = [("latitude", float), ("longitude", float)]


def create_map_projection(configuration_file: str):
	""" create a map projection
	    :param configuration_file: "oceans" | "continents" | "countries"
	"""
	configure = load_options(configuration_file)
	logging.info(f"loaded options from {configuration_file}")
	ф_mesh, λ_mesh, mesh, section_borders = load_mesh(configure["cuts"])
	logging.info(f"loaded a {np.sum(np.isfinite(mesh[:, :, :, 0]))}-node mesh")
	scale_weights = load_pixel_values(configure["scale_weights"], configure["cuts"], mesh.shape[0])
	logging.info(f"loaded the {configure['scale_weights']} map as the area weights")
	shape_weights = load_pixel_values(configure["shape_weights"], configure["cuts"], mesh.shape[0])
	logging.info(f"loaded the {configure['shape_weights']} map as the angle weights")
	width, height = (float(value) for value in configure["size"].split(","))
	logging.info(f"set the maximum map size to {width}×{height} km")

	# assume the coordinates are more or less evenly spaced
	dΦ = EARTH.a*(1 - EARTH.e2)*(1 - EARTH.e2*np.sin(ф_mesh)**2)**(3/2)*(ф_mesh[1] - ф_mesh[0])
	dΛ = EARTH.a*(1 + (1 - EARTH.e2)*np.tan(ф_mesh)**2)**(-1/2)*(λ_mesh[1] - λ_mesh[0])

	# reformat the nodes into a list without gaps or duplicates
	node_indices, node_positions = enumerate_nodes(mesh)
	mesh_indices = Mesh(section_borders, ф_mesh, λ_mesh, node_indices)

	# and then do the same thing for cell corners
	cell_definitions, [cell_shape_weights, cell_scale_weights] = enumerate_cells(
		node_indices, [shape_weights, scale_weights], dΦ, dΛ)

	# set up the fitting constraints that will force the map to fit inside a box
	logging.info(f"projecting section borders...")
	border_matrix = project_section_borders(mesh_indices, 5e-2)
	map_size = np.array([width, height])

	# load the coastline data from Natural Earth
	coastlines = load_coastline_data()

	# now we can bild up the progression schedule
	skeleton_factors = np.ceil(np.geomspace(
		ф_mesh.size/10, 1., int(log2(ф_mesh.size/10)) + 1)).astype(int)
	schedule = [(skeleton_factors[0], 0, False),  # start with the roughest fit, with no bounds
	            (skeleton_factors[0], 0, True)]  # switch to the strict cost function before imposing bounds
	schedule += [(factor, factor, True) for factor in skeleton_factors]  # impose the bounds and then make the mesh finer
	schedule += [(0, 1, True)]  # and finally switch to the complete mesh

	# set up the plotting axes and state variables
	small_fig = plt.figure(figsize=(3, 5), num=f"Elastic {configuration_file} fitting")
	gridspecs = (plt.GridSpec(3, 1, height_ratios=[2, 1, 1]),
	             plt.GridSpec(3, 1, height_ratios=[2, 1, 1], hspace=0))
	hist_axes = small_fig.add_subplot(gridspecs[0][0, :])
	valu_axes = small_fig.add_subplot(gridspecs[1][1, :])
	diff_axes = small_fig.add_subplot(gridspecs[1][2, :], sharex=valu_axes)
	main_fig, map_axes = plt.subplots(figsize=(7, 5), num=f"Elastic {configuration_file}")

	current_state = node_positions
	current_positions = node_positions
	latest_step = np.zeros_like(node_positions)
	values, grads = [], []
	thread_lock = False

	# define the objective functions
	def compute_energy_aggressive(positions: NDArray[float]) -> float:
		# one that aggressively pushes the mesh to have all positive strains
		a, b = compute_principal_strains(restore @ positions,
		                                 cell_definitions, dΦ, dΛ)
		if np.all(a > 0) and np.all(b > 0):
			return -inf
		elif np.any(a < -100) or np.any(b < -100):
			return inf
		else:
			a_term = np.exp(-10*a)
			b_term = np.exp(-10*b)
			return (a_term + b_term).sum()

	def compute_energy_lenient(positions: NDArray[float]) -> float:
		# one that approximates the true cost function without requiring positive strains
		a, b = compute_principal_strains(restore @ positions,
		                                 cell_definitions, dΦ, dΛ)
		scale_term = (a + b - 2)**2
		shape_term = (a - b)**2
		return (scale_term*cell_scale_weights + 2*shape_term*cell_shape_weights).sum()

	def compute_energy_strict(positions: NDArray[float]) -> float:
		# and one that throws an error when any strains are negative
		a, b = compute_principal_strains(restore @ positions,
		                                 cell_definitions, dΦ, dΛ)
		if np.any(a <= 0) or np.any(b <= 0):
			return inf
		else:
			ab = a*b
			scale_term = (ab**2 - 1)/2 - np.log(ab)
			shape_term = (a - b)**2
			return (scale_term*cell_scale_weights + 2*shape_term*cell_shape_weights).sum()

	def record_status(state: NDArray[float], value: float, grad: NDArray[float], step: NDArray[float]) -> None:
		nonlocal current_state, current_positions, latest_step, thread_lock
		while thread_lock: pass
		thread_lock = True
		current_state = state
		current_positions = restore @ state
		latest_step = step
		values.append(value)
		grads.append(np.linalg.norm(grad)*EARTH.R)
		thread_lock = False

	# then minimize! follow the scheduled progression.
	logging.info("begin fitting process.")
	for i, (mesh_factor, bounds_coarseness, final) in enumerate(schedule):
		logging.info(f"fitting pass {i}/{len(schedule)} (coarsened {mesh_factor}x, "
		             f"{'bounded' if bounds_coarseness > 0 else 'unbounded'}, "
		             f"{'final' if final else 'lenient'} cost function)")

		# progress from coarser to finer mesh skeletons
		if mesh_factor > 0:
			gradient_tolerance = 1e-3/EARTH.R
			barrier_tolerance = 1e-2*EARTH.R
			reduce, restore = mesh_skeleton(mesh_indices, mesh_factor)
		else:
			gradient_tolerance = 1e-4/EARTH.R
			barrier_tolerance = 1e-3*EARTH.R
			reduce, restore = Scalar(1), Scalar(1)

		# progress from coarser to finer domain polytopes
		if bounds_coarseness == 0:
			bounds_matrix, bounds_limits = None, inf
		else:
			coarse_border_matrix = border_matrix[np.arange(0, border_matrix.shape[0], bounds_coarseness), :]
			double_border_matrix = SparseNDArray.concatenate([coarse_border_matrix, -coarse_border_matrix])
			bounds_matrix = double_border_matrix @ restore
			bounds_limits = np.array([map_size/2])
			# fit the initial conditions into the bounds each time you impose them
			node_positions = rotate_and_shift(node_positions, *fit_in_rectangle(border_matrix@node_positions))
			border = border_matrix@node_positions
			for k in range(bounds_limits.shape[1]):
				if isfinite(map_size[k]):
					excess = np.ptp(border[:, k])/(2*bounds_limits[0, k])
					set_size = 2*bounds_limits[0, k]*min(1 - .1/λ_mesh.size, 1/excess)
					node_positions[:, k] = interp(node_positions[:, k],
					                              np.min(border[:, k]), np.max(border[:, k]),
					                              -set_size/2, set_size/2)

		node_positions = reduce @ node_positions

		# progress from the quickly-converging approximation to the true cost function
		if not final:
			objective_funcs = [compute_energy_lenient]
		else:
			objective_funcs = [compute_energy_aggressive, compute_energy_strict]

		# each time, run the interior-point with gradient-descent routine
		success = False
		def calculate():
			nonlocal node_positions, success
			for objective_func in objective_funcs:  # for difficult objectives, optimize a short battery of functions
				result = minimize_with_bounds(
					objective_func=objective_func,
					guess=node_positions,
					bounds_matrix=bounds_matrix,
					bounds_limits=bounds_limits,
					report=record_status,
					gradient_tolerance=gradient_tolerance,
					barrier_tolerance=barrier_tolerance)
				node_positions = result.state
				if result.reason == "optimal":  # stopping when you find an optimum
					break
			success = True
		calculation = threading.Thread(target=calculate)
		calculation.start()
		# calculate()
		while True:
			while thread_lock: pass
			thread_lock = True
			done = not calculation.is_alive()
			show_projection(current_state, current_positions,
			                latest_step, values, grads, done,
			                mesh_indices, dΦ, dΛ,
			                cell_definitions, cell_scale_weights,
			                coastlines, border_matrix, width, height,
			                map_axes, hist_axes, valu_axes, diff_axes)
			thread_lock = False
			main_fig.canvas.draw()
			small_fig.canvas.draw()
			plt.pause(2)
			if done: break
		if not success:
			small_fig.canvas.manager.set_window_title("Error!")
			plt.show()
			raise RuntimeError

		# remember to re-mesh the mesh when you're done
		node_positions = restore @ node_positions

	logging.info("end fitting process.")
	small_fig.canvas.manager.set_window_title("Saving...")

	# fit the result in a landscape rectangle
	node_positions = rotate_and_shift(node_positions, *fit_in_rectangle(border_matrix@node_positions))

	# apply the optimized vector back to the mesh
	node_positions_with_nan = np.concatenate([node_positions, [[nan, nan]]])
	mesh = Mesh(section_borders, ф_mesh, λ_mesh,
	            node_positions_with_nan[node_indices, :])

	# save the final version of the plot
	map_axes.axis("off")
	main_fig.savefig(f"../examples/mesh-{configure['number']}.svg",
	                 bbox_inches="tight", pad_inches=0)
	main_fig.savefig(f"../examples/mesh-{configure['number']}.png", dpi=300,
	                 bbox_inches="tight", pad_inches=0)

	# and save the projection itself!
	logging.info("projecting section borders...")
	border_matrix = project_section_borders(mesh_indices, 5e-3)
	logging.info("saving results...")
	save_projection(int(configure["number"]), mesh, configure["section_names"].split(","),
	                decimate_path(border_matrix @ node_positions, resolution=5))

	logging.info(f"projection {configure['number']} saved!")

	small_fig.canvas.manager.set_window_title("Done!")


def enumerate_nodes(mesh: NDArray[float]) -> tuple[NDArray[int], NDArray[float]]:
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


def enumerate_cells(node_indices: NDArray[int], values: list[NDArray[float] | list[NDArray[float]]],
                    dΦ: NDArray[float], dΛ: NDArray[float]) -> tuple[NDArray[int], list[NDArray[float]]]:
	""" take an array of nodes and generate the list of cells in which the elastic energy should be calculated.
	    :param node_indices: the lookup table that tells you the index in the position vector at which is stored
	                         each node at each location in the mesh
	    :param values: a list of the relative importance of the shape of each cell in the cell matrix, for each
	                   section.  it's a list so that you can specify different kinds of importance.
	    :param dΦ: the spacing between each adjacent row of nodes (km)
	    :param dΛ: the spacing between adjacent nodes in each row (km)
	    :return: cell_definitions: the list of cells, each defined by a set of seven indices (the section index, the
	                               two indices specifying its location the matrix, and the indices of the four vertex
	                               nodes (two of them are probably the same node) in the node vector in the order:
	                               west, east, south, north
	             cell_weights: the volume of each cell for elastic-energy-summing porpoises; one 1d array
	                           for each element of values
	"""
	# start off by resampling these in a useful way
	for k in range(len(values)):
		for h in range(node_indices.shape[0]):
			values[k][h] = downsample(values[k][h], node_indices.shape[1:])
		values[k] = np.stack(values[k])

	# assemble a list of all possible cells
	h, i, j = index_grid((node_indices.shape[0],
	                      node_indices.shape[1] - 1,
	                      node_indices.shape[2] - 1))
	h, i, j = h.ravel(), i.ravel(), j.ravel()
	cell_definitions = np.empty((0, 9), dtype=int)
	cell_values = [np.empty((0,), dtype=float)]*len(values)
	for di in range(0, 2):
		for dj in range(0, 2):
			# define them by their indices and neiboring node indices
			west_node = node_indices[h, i + di, j]
			east_node = node_indices[h, i + di, j + 1]
			south_node = node_indices[h, i,     j + dj]
			north_node = node_indices[h, i + 1, j + dj]
			cell_definitions = np.concatenate([
				cell_definitions,
				np.stack([i + di, i + 1 - di, # these first two will get chopd off once I'm done with them
					      h, i + di, j + dj, # these middle three are for generic spacially dependent stuff
				          west_node, east_node, # these bottom four are the really important indices
				          south_node, north_node], axis=-1)])
			for k in range(len(values)):
				cell_values[k] = np.concatenate([
					cell_values[k],
					values[k][h, i, j],
				])

	# then remove all duplicates
	_, unique_indices, final_indices = np.unique(cell_definitions[:, -4:], axis=0, return_index=True, return_inverse=True)
	for k in range(len(values)):
		cell_values[k], _ = np.histogram(final_indices, np.arange(final_indices.max() + 2),
		                                 weights=cell_values[k]) # make sure to add corresponding cell values
	cell_definitions = cell_definitions[unique_indices, :]

	# and remove the ones that rely on missingnodes or that rely on the poles too many times
	missing_node = np.any(cell_definitions[:, -4:] == -1, axis=1)
	degenerate = cell_definitions[:, -4] == cell_definitions[:, -3]
	cell_definitions = cell_definitions[~(missing_node | degenerate), :]
	for k in range(len(values)):
		cell_values[k] = cell_values[k][~(missing_node | degenerate)]

	# you can pull apart the cell definitions now
	cell_node1_is = cell_definitions[:, 0]
	cell_node2_is = cell_definitions[:, 1]
	cell_definitions = cell_definitions[:, 2:]

	# finally, calculate their areas and stuff
	A_1 = dΦ[cell_node1_is]*dΛ[cell_node1_is]
	A_2 = dΦ[cell_node2_is]*dΛ[cell_node2_is]
	cell_areas = (3*A_1 + A_2)/16/(4*np.pi*EARTH.R**2)

	cell_weights = []
	for values in cell_values:
		cell_weights.append(cell_areas*np.minimum(1, np.maximum(MIN_WEIGHT, values)))

	return cell_definitions, cell_weights


def mesh_skeleton(lookup_table: Mesh, factor: int) -> tuple[Tensor, Tensor]:
	""" create a pair of inverse functions that transform points between the full space of possible meshes and a reduced
	    space with fewer degrees of freedom. the idea here is to identify 80% or so of the nodes that can form a
	    skeleton, and from which the remaining nodes can be interpolated. to that end, each parallel will have some
	    number of regularly spaced key nodes, and all nodes adjacent to an edge will be keys, as well, and all nodes
	    that aren't key nodes will be ignored when reducing the position vector and interpolated from neibors when
	    restoring it.
	    :param lookup_table: a Mesh object containing the index of each node's position in the state vector
	    :param factor: approximately how much the resolution should decrease
	    :return: a matrix that linearly reduces a set of node positions to just the bare skeleton, and a matrix that
	             linearly reconstructs the missing node positions from a reduced set.  don't worry about their exact
	             types; they'll both support matrix multiplication with '@'.
	"""
	n_full = np.max(lookup_table.nodes) + 1
	# start by filling out these connection graffs, which are nontrivial because of the layers
	east_neibor = np.full(n_full, -1)
	west_neibor = np.full(n_full, -1)
	north_neibor = np.full(n_full, -1)
	south_neibor = np.full(n_full, -1)
	for h in range(lookup_table.nodes.shape[0]):
		for i in range(lookup_table.nodes.shape[1]):
			for j in range(lookup_table.nodes.shape[2]):
				if lookup_table.nodes[h, i, j] != -1:
					if j - 1 >= 0 and lookup_table.nodes[h, i, j - 1] != -1:
						west_neibor[lookup_table.nodes[h, i, j]] = lookup_table.nodes[h, i, j - 1]
					if j + 1 < lookup_table.nodes.shape[2] and lookup_table.nodes[h, i, j + 1] != -1:
						east_neibor[lookup_table.nodes[h, i, j]] = lookup_table.nodes[h, i, j + 1]
					if i - 1 >= 0 and lookup_table.nodes[h, i - 1, j] != -1:
						south_neibor[lookup_table.nodes[h, i, j]] = lookup_table.nodes[h, i - 1, j]
					if i + 1 < lookup_table.nodes.shape[1] and lookup_table.nodes[h, i + 1, j] != -1:
						north_neibor[lookup_table.nodes[h, i, j]] = lookup_table.nodes[h, i + 1, j]

	# then decide which nodes should be independently defined in the skeleton
	has_defined_neibors = np.full(n_full + 1, False) # (this array has an extra False at the end so that -1 works nicely)
	is_defined = np.full(n_full, False)
	# start by marking some evenly spaced interior points
	if factor >= 1.5:
		num_ф = max(3, floor((lookup_table.nodes.shape[1] - 1)/factor))
		important_ф = np.linspace(-90, 90, num_ф, endpoint=False)
		important_i = np.round((important_ф + 90)*(lookup_table.nodes.shape[1] - 1)/180)
	else:
		important_i = np.arange(lookup_table.nodes.shape[1])
	for h in range(lookup_table.nodes.shape[0]):
		for i in range(lookup_table.nodes.shape[1]):
			num_λ = max(1, round((lookup_table.nodes.shape[2] - 1)*np.cos(lookup_table.ф[i])/factor))
			if num_λ <= lookup_table.nodes.shape[2]/1.5:
				important_λ = np.linspace(0, 360, num_λ, endpoint=False)
				important_j = np.round(important_λ*(lookup_table.nodes.shape[2] - 1)/360)
			else:
				important_j = np.arange(lookup_table.nodes.shape[2])
			for j in range(lookup_table.nodes.shape[2]):
				if lookup_table.nodes[h, i, j] != -1:
					important_row = i in important_i
					important_col = j in important_j
					has_defined_neibors[lookup_table.nodes[h, i, j]] |= important_row
					is_defined[lookup_table.nodes[h, i, j]] |= important_col
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
	n_weit, s_weit = get_interpolation_weights(n_distance, s_distance)
	ne_weit, nw_weit = get_interpolation_weights(ne_distance, nw_distance)
	se_weit, sw_weit = get_interpolation_weights(se_distance, sw_distance)
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
	reduction = SparseNDArray.identity(n_full)[is_defined, :]
	restoration = SparseNDArray.from_coordinates(
		[n_partial], np.expand_dims(reindex[defining_indices], axis=-1), defining_weits)
	return reduction, restoration


def follow_graph(progression: NDArray[int], frum: NDArray[int], until: NDArray[float]) -> tuple[NDArray[int], NDArray[int]]:
	""" take with a markov-chain-like-thing in the form of an array of indices and follow it to some conclusion.
	    :param progression: the indices of the nodes that follow from each node
	    :param frum: the starting state, an array of indices
	    :param until: a boolean array indicating points that don't need to follow to the next part of the graph
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


def get_interpolation_weights(distance_a: NDArray[int], distance_b: NDArray[int]) -> tuple[NDArray[float], NDArray[float]]:
	""" compute the weits needed to linearly interpolate a point between two fixed points,
	    given the distance of each respective reference to the point of interpolation.
	"""
	weits_a = np.empty(distance_a.shape)
	normal = distance_a + distance_b != 0
	weits_a[normal] = distance_b[normal]/(distance_a + distance_b)[normal]
	weits_a[~normal] = 1
	weits_b = 1 - weits_a
	return weits_a, weits_b


def compute_principal_strains(positions: NDArray[float],
                              cell_definitions: NDArray[int],
                              dΦ: NDArray[float], dΛ: NDArray[float]
                              ) -> tuple[NDArray[float], NDArray[float]]:
	""" take a set of cell definitions and 2D coordinates for each node, and calculate
	    the Tissot-ellipse semiaxes of each cell.
	    :param positions: the vector specifying the location of each node in the map plane
	    :param cell_definitions: the list of cells, each defined by seven indices
	    :param dΦ: the distance between adjacent rows of nodes (km)
	    :param dΛ: the distance between adjacent nodes in each row (km)
	    :return: the major primary strains, and the minor primary strains
	"""
	i = cell_definitions[:, 1]

	west = positions[cell_definitions[:, 3], :]
	east = positions[cell_definitions[:, 4], :]
	F_λ = ((east - west)/dΛ[i, np.newaxis])
	dxdΛ, dydΛ = F_λ[:, 0], F_λ[:, 1]

	south = positions[cell_definitions[:, 5], :]
	north = positions[cell_definitions[:, 6], :]
	F_ф = ((north - south)/dΦ[i, np.newaxis])
	dxdΦ, dydΦ = F_ф[:, 0], F_ф[:, 1]

	trace = np.sqrt((dxdΛ + dydΦ)**2 + (dxdΦ - dydΛ)**2)/2
	antitrace = np.sqrt((dxdΛ - dydΦ)**2 + (dxdΦ + dydΛ)**2)/2
	return trace + antitrace, trace - antitrace


def show_projection(fit_positions: NDArray[float], all_positions: NDArray[float], velocity: NDArray[float],
                    values: list[float], grads: list[float], final: bool,
                    mesh_indices: Mesh, dΦ: NDArray[float], dΛ: NDArray[float],
                    cell_definitions: NDArray[int], cell_weights: NDArray[float],
                    coastlines: list[np.array], border: NDArray[float] | SparseNDArray,
                    map_width: float, map_hite: float,
                    map_axes: plt.Axes, hist_axes: plt.Axes,
                    valu_axes: plt.Axes, diff_axes: plt.Axes) -> None:
	""" display the current state of the optimization process, including a preliminary map, a
	    distortion histogram, and a convergence as a function of time plot
	"""
	map_axes.clear()
	mesh = Mesh(mesh_indices.section_borders, mesh_indices.ф, mesh_indices.λ,
	            np.where(mesh_indices.nodes[:, :, :, np.newaxis] != -1,
	                     all_positions[mesh_indices.nodes, :], nan))
	for h in range(mesh.num_sections):
		# plot the underlying mesh_indices for each section
		map_axes.plot(mesh.nodes[h, :, :, 0], mesh.nodes[h, :, :, 1], "#bbb", linewidth=.3, zorder=1)
		map_axes.plot(mesh.nodes[h, :, :, 0].T, mesh.nodes[h, :, :, 1].T, "#bbb", linewidth=.3, zorder=1)
		# crudely project and plot the coastlines onto each section
		project = RegularGridInterpolator([mesh.ф, mesh.λ], mesh.nodes[h, :, :, :],
		                                  bounds_error=False, fill_value=nan)
		for line in coastlines:
			projected_line = project(line)
			map_axes.plot(projected_line[:, 0], projected_line[:, 1], "#000", linewidth=.8, zorder=2)

	# plot the outline of the mesh_indices
	border_points = border @ all_positions
	map_axes.fill(border_points[:, 0], border_points[:, 1],
	              facecolor="none", edgecolor="#000", linewidth=1.3, zorder=2)
	# plot the bounding rectangle if there is one
	if not final:
		map_axes.plot(np.multiply([-1, 1, 1, -1, -1], map_width/2),
		              np.multiply([-1, -1, 1, 1, -1], map_hite/2), "#000", linewidth=.3, zorder=2)

	if not final and velocity is not None:
		# indicate the speed of each node
		map_axes.scatter(fit_positions[:, 0], fit_positions[:, 1], s=5,
		                 c=-np.linalg.norm(velocity, axis=1),
		                 vmax=0, cmap=CUSTOM_CMAP["speed"], zorder=0)

	a, b = compute_principal_strains(all_positions, cell_definitions, dΦ, dΛ)

	# mark any nodes with nonpositive principal strains
	worst_cells = np.nonzero((a <= 0) | (b <= 0))[0]
	for cell in worst_cells:
		h, i, j, east, west, north, south = cell_definitions[cell, :]
		map_axes.plot(all_positions[[east, west], 0], all_positions[[east, west], 1], "#f50", linewidth=.8)
		map_axes.plot(all_positions[[north, south], 0], all_positions[[north, south], 1], "#f50", linewidth=.8)
	map_axes.axis("equal")
	map_axes.margins(.01)

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
	valu_axes.set_xlim(len(values) - 100, len(values))
	valu_axes.set_ylim(0, 6*values[-1] if values else 1)
	valu_axes.minorticks_on()
	valu_axes.yaxis.set_tick_params(which='both')
	valu_axes.grid(which="both", axis="y")

	# plot the convergence criteria over time
	diff_axes.clear()
	diff_axes.scatter(np.arange(len(grads)), grads, s=2, zorder=11)
	ylim = max(2e-2, np.min(grads, initial=1e3)*5e2)
	diff_axes.set_ylim(ylim/1e3, ylim)
	diff_axes.set_yscale("log")
	diff_axes.grid(which="major", axis="y")


def save_projection(number: int, mesh: Mesh, section_names: list[str],
                    projected_border: NDArray[float]) -> None:
	""" save all of the important map projection information as a HDF5 and text file.
	    :param number: the sequence number of this map projection
	    :param mesh: the mesh of the projection being saved
	    :param section_names: a list of the names of the n sections. these will be added to the HDF5 file as attributes.
	    :param projected_border: the px2 array of cartesian points representing the border of the whole map
	"""
	assert len(section_names) == mesh.num_sections

	with open("../resources/lang.json", "r", encoding="utf-8") as f:
		languages = json.load(f)

	# start by calculating some things
	((left, bottom), (right, top)) = get_bounding_box(projected_border)
	raster_resolution = 20
	x_raster = np.linspace(left, right, raster_resolution)
	y_raster = np.linspace(bottom, top, raster_resolution)
	inverse_raster = np.degrees(inverse_project(
		np.transpose(np.meshgrid(x_raster, y_raster, indexing="ij"), (1, 2, 0)), mesh))

	# do the self-explanatory HDF5 file
	for language_code, lang in languages.items():
		language_suffix = f"-{language_code}" if language_code != "en" else ""
		numeral = lang["numerals"][number]
		h5_xy_tuple = [(lang["x"], float), (lang["y"], float)]
		h5_фλ_tuple = [(lang["latitude"], float), (lang["longitude"], float)]

		with h5py.File(f"../projection/{lang['elastic-earth']}-{numeral}{language_suffix}.h5", "w") as file:
			file.attrs[lang["name"]] = lang["elastic earth #"].format(numeral)
			file.attrs[lang["descript"]] = lang[f"descript{number}"]
			file.attrs[lang["num sections"]] = mesh.num_sections
			file.create_dataset(lang["projected border"], shape=(projected_border.shape[0],), dtype=h5_xy_tuple)
			file[lang["projected border"]][lang["x"]] = projected_border[:, 0]
			file[lang["projected border"]][lang["y"]] = projected_border[:, 1]
			file.create_dataset(lang["bounding box"], shape=(2,), dtype=h5_xy_tuple)
			file[lang["bounding box"]][lang["x"]] = [left, right]
			file[lang["bounding box"]][lang["y"]] = [bottom, top]
			file[lang["bounding box"]].attrs[lang["units"]] = "km"
			file[lang["sections"]] = [lang["section #"].format(h) for h in range(mesh.num_sections)]
			group = file.create_group(lang["inverse"])
			group[lang["x"]] = x_raster
			group[lang["x"]].attrs[lang["units"]] = "km"
			group[lang["x"]].make_scale()
			group[lang["y"]] = y_raster
			group[lang["y"]].attrs[lang["units"]] = "km"
			group[lang["y"]].make_scale()
			group.create_dataset(lang["inverse"], shape=inverse_raster.shape[:2], dtype=h5_фλ_tuple)
			group[lang["inverse"]][lang["latitude"]] = inverse_raster[:, :, 0]
			group[lang["inverse"]][lang["longitude"]] = inverse_raster[:, :, 1]
			group[lang["inverse"]].attrs[lang["units"]] = "°"
			group[lang["inverse"]].dims[0].attach_scale(group[lang["x"]])
			group[lang["inverse"]].dims[1].attach_scale(group[lang["y"]])

			for h in range(mesh.num_sections):
				group = file.create_group(lang["section #"].format(h))
				group.attrs[lang["name"]] = lang[section_names[h]]
				group[lang["latitude"]] = np.degrees(mesh.ф)
				group[lang["latitude"]].attrs[lang["units"]] = "°"
				group[lang["latitude"]].make_scale()
				group[lang["longitude"]] = np.degrees(mesh.λ)
				group[lang["longitude"]].make_scale()
				group[lang["longitude"]].attrs[lang["units"]] = "°"
				group.create_dataset(lang["projected points"], shape=(mesh.ф.size, mesh.λ.size), dtype=h5_xy_tuple)
				group[lang["projected points"]][lang["x"]] = mesh.nodes[h, :, :, 0]
				group[lang["projected points"]][lang["y"]] = mesh.nodes[h, :, :, 1]
				group[lang["projected points"]].attrs[lang["units"]] = "km"
				group[lang["projected points"]].dims[0].attach_scale(group[lang["latitude"]])
				group[lang["projected points"]].dims[1].attach_scale(group[lang["longitude"]])
				group.create_dataset(lang["border"], shape=(mesh.section_borders[h].shape[0],), dtype=h5_фλ_tuple)
				group[lang["border"]][lang["latitude"]] = np.degrees(mesh.section_borders[h][:, 0])
				group[lang["border"]][lang["longitude"]] = np.degrees(mesh.section_borders[h][:, 1])
				group[lang["border"]].attrs[lang["units"]] = "°"
				((left_h, bottom_h), (right_h, top_h)) = get_bounding_box(mesh.nodes[h, :, :, :])
				group.create_dataset(lang["bounding box"], shape=(2,), dtype=h5_xy_tuple)
				group[lang["bounding box"]][lang["x"]] = [max(left, left_h), min(right, right_h)]
				group[lang["bounding box"]][lang["y"]] = [max(bottom, bottom_h), min(top, top_h)]
				group[lang["bounding box"]].attrs[lang["units"]] = "km"

		# then save a simpler but larger and less explanatory txt file
		with open(f"../projection/{lang['elastic-earth']}-{numeral}{language_suffix}.csv", "w", encoding="utf-8") as f:
			f.write(lang["projection header"].format(numeral, mesh.num_sections)) # the number of sections
			for h in range(mesh.num_sections):
				f.write(lang["section header"].format(h))
				f.write(lang["section border header"].format(mesh.section_borders[h].shape[0])) # the number of section border vertices
				for i in range(mesh.section_borders[h].shape[0]):
					f.write(f"{mesh.section_borders[h][i, 0]:5.1f},{mesh.section_borders[h][i, 1]:5.1f}\n") # the section border vertices (°)
				f.write(lang["section points header"].format(*mesh.nodes[h].shape)) # the shape of the section mesh_indices
				for i in range(mesh.nodes.shape[1]):
					for j in range(mesh.nodes.shape[2]):
						f.write(f"{mesh.nodes[h, i, j, 0]:8.2f},{mesh.nodes[h, i, j, 1]:8.2f}") # the section mesh_indices points (km)
						if j != mesh.nodes.shape[2] - 1:
							f.write(", ")
					f.write("\n")
			f.write(lang["border header"].format(projected_border.shape[0])) # the number of map edge vertices
			for i in range(projected_border.shape[0]):
				f.write(f"{projected_border[i, 0]:8.2f},{projected_border[i, 1]:8.2f}\n") # the map edge vertices (km)
			f.write(lang["inverse header"].format(*inverse_raster.shape)) # the shape of the sample raster
			f.write(f"{left} - {right}, {bottom} - {top}\n") # the bounding box of the sample raster
			for j in range(inverse_raster.shape[1]):
				for i in range(inverse_raster.shape[0]):
					f.write(f"{inverse_raster[i, j, 0]:5.1f},{inverse_raster[i, j, 1]:5.1f}") # the sample raster (°)
					if j != inverse_raster.shape[1] - 1:
						f.write(", ")
				f.write("\n")


def load_options(filename: str) -> dict[str, str]:
	""" load a simple colon-separated text file """
	options = dict()
	with open(f"../spec/options_{filename}.txt", "r", encoding="utf-8") as file:
		for line in file.readlines():
			key, value = line.split(":")
			options[key.strip()] = value.strip()
	return options


def load_pixel_values(filename: str, cut_set: str, num_sections: int) -> list[NDArray[float]]:
	""" load and resample a generic 2D raster image """
	if filename == "uniform":
		return [np.array(1.)]*num_sections
	else:
		values = []
		for h in range(num_sections):
			values.append(tifffile.imread(f"../spec/pixels_{cut_set}_{h}_{filename}.tif"))
		return values


def load_coastline_data(reduction=2) -> list[NDArray[float]]:
	coastlines = []
	with shapefile.Reader(f"../data/ne_110m_coastline.zip") as shapef:
		for shape in shapef.shapes():
			if len(shape.points) > 3*reduction:
				coastlines.append(np.radians(shape.points)[::reduction, ::-1])
	return coastlines


def load_mesh(filename: str) -> tuple[NDArray[float], NDArray[float], NDArray[float], list[NDArray[float]]]:
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


def project(points: list[tuple[float, float]] | NDArray[float], mesh: Mesh,
            section_index: int = None) -> SparseNDArray | NDArray[float]:
	""" take some points, giving all sections of the map projection, and project them into the plane, representing them
	    either with their resulting cartesian coordinates or as matrix that multiplies by the vector of node positions
	    to produce an array of points.  you can either specify a section_index, in which case all points will be
	    projected to a single section (and points outide of that section’s bounds will be projected as NaN), or not
	    specify it, in which case all points will be projected to the one section for which they’re inside its borders.
	    :param points: an m×...×n×2 array of the spherical coordinates of the points to project (radians)
	    :param mesh: the mesh onto which to project the points
	    :param section_index: the index of the section to use for all points
	"""
	if mesh.section_borders is None and section_index is None:
		raise ValueError("at least one of section_borders and section_index must not be none.")

	positions_known = mesh.nodes.ndim == 4

	# first, we set up an identity matrix of sorts, if the positions aren't currently known
	if positions_known:
		valid = np.isfinite(mesh.nodes[:, :, :, 0])
	else:
		valid = mesh.nodes != -1
		num_nodes = np.max(mesh.nodes) + 1
		new_nodes = SparseNDArray.concatenate([
			SparseNDArray.identity(num_nodes),
			SparseNDArray.zeros((1, num_nodes), 1)]
		).to_array_array()[mesh.nodes]
		mesh = Mesh(mesh.section_borders, mesh.ф, mesh.λ, new_nodes)

	# next, we must identify any nodes that exist in multiple layers
	hs = np.arange(mesh.num_sections)
	shared = np.tile(
		np.any(
			valid & np.all(
				np.reshape(mesh.nodes[hs, ...] == mesh.nodes[hs - 1, ...], mesh.nodes.shape[:3] + (-1,)),
				axis=3
			), axis=0
		), (mesh.num_sections, 1, 1))

	# and start calculating gradients
	ф_gradients = np.empty(mesh.nodes.shape, dtype=mesh.nodes.dtype)
	λ_gradients = np.empty(mesh.nodes.shape, dtype=mesh.nodes.dtype)
	for where in [valid, shared]:
		gradients, gradients_mask = gradient(mesh.nodes, mesh.ф, where=where, axis=1)
		ф_gradients[~gradients_mask, ...] = gradients[~gradients_mask, ...]
		gradients, gradients_mask = gradient(mesh.nodes, mesh.λ, where=where, axis=2)
		λ_gradients[~gradients_mask, ...] = gradients[~gradients_mask, ...]

	# finally, interpolate
	result = []
	for point in points:
		h = section_index
		if h is None:
			for trial_h, border in zip(hs, mesh.section_borders): # on the correct section
				if inside_region(*point, border, period=2*pi):
					h = trial_h
					break
		result.append(smooth_interpolate(point, (mesh.ф, mesh.λ), mesh.nodes[h, ...],
		                                 (ф_gradients[h, ...], λ_gradients[h, ...])))

	# convert to the correct type
	if positions_known:
		return np.array(result)
	else:
		return SparseNDArray.concatenate(result)


def inverse_project(points: NDArray[float], mesh: Mesh) -> SparseNDArray | NDArray[float]:
	""" take some points, specifying a section of the map projection, and project them from the plane back to the globe,
	    representing the result as the resulting latitudes and longitudes
	    :param points: the m×...×n×2 array of planar coordinates of the points to project (km)
	    :param mesh: the mesh on which to find the given points
	    :return: the m×...×n×2 array of spherical coordinates corresponding to the input points (radians)
	"""
	hs = range(mesh.num_sections)

	# do each point one at a time, since this doesn't need to be super fast
	result = np.full(points.shape, nan)
	for point_index, point in enumerate(points.reshape((-1, 2))):
		point_index = np.unravel_index(point_index, points.shape[:-1])
		logging.info(f"{point_index}/{points.shape[:-1]}")
		possible_results = np.empty((mesh.num_sections, 2), dtype=float)
		closenesses = np.empty(mesh.num_sections, dtype=float)
		for h in hs:
			# start by vectorizedly scanning every node position
			residuals = np.sum((mesh.nodes[h, :, :, :] - np.array([point]))**2, axis=-1)
			closest_node = np.unravel_index(np.nanargmin(residuals.ravel()), residuals.shape)
			ф_closest, λ_closest = mesh.nodes[(h, *closest_node)]
			# this scan minimization will serve as the backup result if we find noting better
			possible_results[h, :] = [ф_closest, λ_closest]
			closenesses[h] = residuals[closest_node]
			# but more importantly, this will serve as the initial guess for the more detailed search
			i_guess = bin_index(ф_closest, mesh.ф)
			j_guess = bin_index(λ_closest, mesh.λ)
			# look at each _cell_ in the vicinity and see if any contain the point
			for i, j in search_out_from(i_guess, j_guess, mesh.nodes.shape[1:3], 6):
				# look for a cell that contains the point
				if np.all(np.isfinite(mesh.nodes[i-1:i+1, j-1:j+1])):
					sw = mesh.nodes[h, i - 1, j - 1]
					se = mesh.nodes[h, i - 1, j]
					nw = mesh.nodes[h, i, j - 1]
					ne = mesh.nodes[h, i, j]
					if inside_polygon(*point, np.array([ne, nw, sw, se]), convex=True):
						# do inverse 2d linear interpolation (it's harder than one mite expect!)
						coords = [nan, nan]
						things = [(0, mesh.ф, i, [[sw, se], [nw, ne]]), (1, mesh.λ, j, [[sw, nw], [se, ne]])]
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
						possible_results[h, :] = coords
						closenesses[h] = 0
						break

		# crudely deal with multiple possible projections for each point
		# first prioritize having an inverse projection near the correct anser
		best_h = np.argmin(closenesses)
		result[point_index] = possible_results[best_h]
		# but mainly prioritize being inside the section borders
		for h in hs:
			if inside_region(possible_results[h, 0], possible_results[h, 1],
			                 mesh.section_borders[h], period=2*pi):
				result[point_index] = possible_results[h]

	return result


def smooth_interpolate(xs: Sequence[float | NDArray[float]], x_grids: Sequence[NDArray[float]],
                       z_grid: NDArray[float] | SparseNDArray, dzdx_grids: Sequence[NDArray[float] | SparseNDArray],
                       differentiate=None) -> float | NDArray[float] | SparseNDArray:
	""" perform a bi-cubic interpolation of some quantity on a grid, using precomputed gradients
	    :param xs: the location vector that specifies where on the grid we want to look
	    :param x_grids: the x values at which the output is defined (each x_grid should be 1d)
	    :param z_grid: the known values at the grid intersection points
	    :param dzdx_grids: the known derivatives with respect to ф at the grid intersection points
	    :param differentiate: if set to a nonnegative number, this will return not the interpolated value at the given
	                          point, but the interpolated *derivative* with respect to one of the input coordinates. the
	                          value of the parameter indexes which coordinate along which to index
	    :return: the value z that fits at xs
	"""
	if len(xs) != len(x_grids) or len(x_grids) != len(dzdx_grids):
		raise ValueError("the number of dimensions is not consistent")
	ndim = len(xs)
	item_ndim = z_grid.ndim - ndim if type(z_grid) is np.ndarray else 1

	# choose a cell in the grid
	key = [np.interp(x, x_grid, np.arange(x_grid.size)) for x, x_grid in zip(xs, x_grids)]

	# calculate the cubic-interpolation weights for the adjacent values and gradients
	value_weights, slope_weights = [], []
	for k, i_full in enumerate(key):
		i = key[k] = np.minimum(np.floor(i_full).astype(int), x_grids[k].size - 2)
		ξ, ξ2, ξ3 = i_full - i, (i_full - i)**2, (i_full - i)**3
		dx = x_grids[k][i + 1] - x_grids[k][i]
		if differentiate is None or differentiate != k:
			value_weights.append([2*ξ3 - 3*ξ2 + 1, -2*ξ3 + 3*ξ2])
			slope_weights.append([(ξ3 - 2*ξ2 + ξ)*dx, (ξ3 - ξ2)*dx])
		else:
			value_weights.append([(6*ξ2 - 6*ξ)/dx, (-6*ξ2 + 6*ξ)/dx])
			slope_weights.append([3*ξ2 - 4*ξ + 1, 3*ξ2 - 2*ξ])
	value_weights = np.meshgrid(*value_weights, indexing="ij", sparse=True)
	slope_weights = np.meshgrid(*slope_weights, indexing="ij", sparse=True)

	# get the indexing all set up correctly
	index = tuple(np.meshgrid(*([i, i + 1] for i in key), indexing="ij", sparse=True))
	full = (slice(None),)*ndim + (np.newaxis,)*item_ndim

	# then multiply and combine all the things
	weits = product(value_weights)[full]
	result = np.sum(weits*z_grid[index], axis=tuple(range(ndim)))
	for k in range(ndim):
		weits = product(value_weights[:k] + [slope_weights[k]] + value_weights[k+1:])[full]
		result += np.sum(weits*dzdx_grids[k][index], axis=tuple(range(ndim)))
	return result


def gradient(Y: NDArray[float], x: NDArray[float], where: NDArray[bool], axis: int) -> (NDArray[float], NDArray[bool]):
	""" Return the gradient of an N-dimensional array.
	    The gradient is computed using second ordre accurate central differences in the interior points and either
	    first- or twoth-order accurate one-sides (forward or backwards) differences at the boundaries. The returned
	    gradient hence has the same shape as the input array.
	    :param Y: the values of which to take the gradient
	    :param x: the values by which we take the gradient in a 1d array whose length matches Y.shape[axis]
	    :param where: the values that are valid and should be consulted when calculating this gradient. it should have
	                  the same shape as Y. values of Y corresponding to a falsy in where will be ignored, and gradients
	                  centerd at such points will be markd as nan.
	    :param axis: the axis along which to take the gradient
	    :return: an array with the gradient values, as well as a boolean mask indicating which of the returnd values are
	             invalid (because there were not enuff valid inputs in the vicinity)
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


def project_section_borders(index_mesh: Mesh,
                            resolution: float) -> SparseNDArray:
	""" take the section borders, concatenate them, project them, and trim off the shared edges """
	borders = []
	for h, border in enumerate(index_mesh.section_borders): # take the border of each section
		first_pole = np.nonzero(abs(border[:, 0]) == pi/2)[0][0]
		border = np.concatenate([border[first_pole:], border[1:first_pole + 1]]) # rotate the path so it starts and ends at a pole
		border = border[dilate(abs(border[:, 0]) != pi/2, 1)] # and then remove points that move along the pole
		borders.append(project(refine_path(border, resolution, period=2*pi),  # finally, refine it before projecting
		                       index_mesh, section_index=h))
	border_matrix = simplify_path(SparseNDArray.concatenate(borders), cyclic=True) # simplify the borders together to remove excess
	return border_matrix


def downsample(full: NDArray[float], shape: tuple):
	""" decrease the size of a numpy array by setting each pixel to the mean of the pixels
	    in the original image for which it was the nearest neibor
	"""
	if full.shape == ():
		return np.full(shape, full)
	assert len(shape) == len(full.shape)
	for i in range(len(shape)):
		assert shape[i] < full.shape[i]
	reduce = np.empty(shape)
	i_reduce = (np.arange(full.shape[0])/full.shape[0]*reduce.shape[0]).astype(int)
	j_reduce = (np.arange(full.shape[1])/full.shape[1]*reduce.shape[1]).astype(int)
	for i in range(shape[0]):
		for j in range(shape[1]):
			reduce[i][j] = np.mean(full[i_reduce == i][:, j_reduce == j])
	return reduce


def get_bounding_box(points: NDArray[float]) -> NDArray[float]:
	""" compute the maximum and minimums of this set of points and package it as [[left, bottom], [right, top]] """
	return np.array([
		[np.nanmin(points[..., 0]), np.nanmin(points[..., 1])],
		[np.nanmax(points[..., 0]), np.nanmax(points[..., 1])],
	])


def product(values: Iterable[NDArray[float] | float | int]) -> NDArray[float] | float | int:
	result = None
	for value in values:
		if result is None:
			result = value
		else:
			result = result*value
	return result


class Mesh:
	def __init__(self, section_borders: list[NDArray[float]],
	             ф: NDArray[float], λ: NDArray[float], nodes: NDArray[float] | NDArray[int]):
		""" a record containing the arrays that define an arbitrary map projection
		    :param ф: the m latitudes at which the nodes are defined (radians)
		    :param λ: the n longitudes at which the nodes are defined (radians)
		    :param nodes: either a) an l×m×n×2 array of x and y coordinates for each section, from which the
		                            coordinates of the projected points will be determined (km), or
	                             b) a l×m×n array of node indices for each section, indicating that the result
	                                should be a matrix that you can multiply by the node positions later.
		    :param section_borders: the l paths defining each section given as o×2 arrays (radians)
		"""
		assert nodes.shape[:3] == (len(section_borders), ф.size, λ.size)
		self.section_borders = section_borders
		self.ф = ф
		self.λ = λ
		self.nodes = nodes
		self.num_sections = len(section_borders)


if __name__ == "__main__":
	create_map_projection("continents")
	create_map_projection("oceans")
	create_map_projection("countries")

	plt.show()
