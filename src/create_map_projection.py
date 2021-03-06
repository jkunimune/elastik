#!/usr/bin/env python
"""
create_map_projection.py

take a basic mesh and optimize it according to a particular cost function in order to
create a new Elastic Projection.
"""
from typing import Callable

import h5py
import numpy as np
import shapefile
import tifffile
from matplotlib import pyplot as plt
from scipy import interpolate

from cmap import CUSTOM_CMAP
from optimize import minimize, GradientSafe
from util import dilate, h5_str, EARTH


CONFIGURATION_FILE = "oceans" # "continents"; "countries"


def get_bounding_box(points: np.ndarray) -> np.ndarray:
	""" compute the maximum and minimums of this set of points and package it as
	    [[left, bottom], [right, top]]
	"""
	return np.array([
		[np.nanmin(points[..., 0]), np.nanmin(points[..., 1])],
		[np.nanmax(points[..., 0]), np.nanmax(points[..., 1])], # TODO: account for the border, and for spline interpolation
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


def find_or_add(vector: np.ndarray, vectors: list[np.ndarray]) -> tuple[bool, int]:
	""" add vector to vectors if it's not already there, then return its index
	    :return: whether we had to add it to the list (because we didn't find it), and
	             the index of this item in the list
	"""
	if np.any(np.isnan(vector)):
		return False, -1
	else:
		for k in range(len(vectors)):
			if np.all(vectors[k] == vector):
				return False, k
		vectors.append(vector)
		return True, len(vectors) - 1


def enumerate_nodes(mesh: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
	""" take an array of positions for a mesh and generate the list of unique nodes,
	    returning mappings from the old mesh to the new indices and the n??2 list positions
	"""
	node_indices = np.empty(mesh.shape[:3], dtype=int)
	node_positions = []
	for h in range(mesh.shape[0]):
		for i in range(mesh.shape[1]):
			for j in range(mesh.shape[2]):
				_, node_indices[h, i, j] = find_or_add(mesh[h, i, j, :], node_positions)
	return node_indices, np.array(node_positions)


def enumerate_cells(??: np.ndarray, node_indices: np.ndarray, values: np.ndarray,
                    ) -> tuple[np.ndarray, np.ndarray]:
	""" take an array of nodes and generate the list of cells in which the elastic energy
	    should be calculated.
	"""
	cell_definitions = []
	cell_weights = []
	for h in range(node_indices.shape[0]):
		for i in range(node_indices.shape[1] - 1):
			for j in range(node_indices.shape[2] - 1):
				# each corner of each cell should be represented separately
				for di in range(0, 2):
					for dj in range(0, 2):
						if i + di != 0 and i + di != node_indices.shape[1] - 1: # skip the corners next to the poles
							# define each cell corner by its i and j, and the indices of its adjacent nodes
							west_node = node_indices[h, i + di, j]
							east_node = node_indices[h, i + di, j + 1]
							south_node = node_indices[h, i,     j + dj]
							north_node = node_indices[h, i + 1, j + dj]
							cell_definition = np.array([h, i + di, j + dj,
							                            west_node, east_node,
							                            south_node, north_node])

							if not np.any(cell_definition == -1):
								added, _ = find_or_add(cell_definition, cell_definitions)
								if added:
									# calculate the area of the corner when appropriate
									??_1 = ??[i + di]
									??_2 = ??[i + 1 - di]
									area = d??*d??*(3*np.cos(??_1) + np.cos(??_2))/16/(4*np.pi)
									value = values[i, j]
									cell_weights.append(area*value)

	return np.array(cell_definitions), np.array(cell_weights)


def mesh_skeleton(??: np.ndarray, lookup_table: np.ndarray
                  ) -> tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]:
	""" create a pair of inverse functions that transform points between the full space of
	    possible meshes and a reduced space with fewer degrees of freedom. the idea here
	    is to identify 80% or so of the nodes that can form a skeleton, and from which the
	    remaining nodes can be interpolated. to that end, each parallel will have some
	    number of regularly spaced key nodes, and all nodes adjacent to an edge will
	    be keys, as well, and all nodes that aren't key nodes will be ignored when
	    reducing the position vector and interpolated from neibors when restoring it.
	    :param ??: the latitudes of the nodes
	    :param lookup_table: the index of each node's position in the state vector
	    :return: a function to linearly reduce a set of node positions to just the bare
	             skeleton, and a function to linearly reconstruct the missing node
	             positions from a reduced set
	"""
	n_full = np.max(lookup_table) + 1
	# start by filling out these connection graffs, which are nontrivial because of the layers
	left_neibor = np.full(n_full, -1)
	rite_neibor = np.full(n_full, -1)
	for h in range(lookup_table.shape[0]):
		for i in range(lookup_table.shape[1]):
			for j in range(lookup_table.shape[2]):
				if lookup_table[h, i, j] != -1:
					if j - 1 >= 0 and lookup_table[h, i, j - 1] != -1:
						left_neibor[lookup_table[h, i, j]] = lookup_table[h, i, j - 1]
					if j + 1 < lookup_table.shape[2] and lookup_table[h, i, j + 1] != -1:
						rite_neibor[lookup_table[h, i, j]] = lookup_table[h, i, j + 1]

	# then decide which nodes should be independently defined in the skeleton
	important = (left_neibor == -1) | (rite_neibor == -1) # points at the edge of a cut
	important[lookup_table[:, :, 0]] |= (lookup_table[:, :, 0] != -1) # the defined portion of the left edge
	important[lookup_table[:, :, -1]] |= (lookup_table[:, :, -1] != -1) # the defined portion of the right edge
	for h in range(lookup_table.shape[0]):
		for i in range(lookup_table.shape[1]):
			period = int(round(1/np.cos(??[i])))
			for j in range(lookup_table.shape[2]):
				if lookup_table[h, i, j] != -1:
					if j%period == 0:
						important[lookup_table[h, i, j]] = True

	# then decide how to define the ones that aren't
	defining_indices = np.empty((n_full, 2), dtype=int)
	defining_weits = np.empty((n_full, 2), dtype=float)
	for k0 in range(n_full):
		# important nodes are defined by the corresponding row in the reduced vector
		if important[k0]:
			defining_indices[k0, :] = np.sum(important[:k0])
			defining_weits[k0, 0] = 1
		# each remaining node is a linear combination of two important nodes
		else:
			k_left, distance_left = k0, 0
			while not important[k_left]:
				k_left = left_neibor[k_left]
				distance_left += 1
			k_rite, distance_rite = k0, 0
			while not important[k_rite]:
				k_rite = rite_neibor[k_rite]
				distance_rite += 1
			defining_indices[k0, 0] = np.sum(important[:k_left])
			defining_indices[k0, 1] = np.sum(important[:k_rite])
			defining_weits[k0, 0] = distance_rite/(distance_left + distance_rite)
	defining_weits[:, 1] = 1 - defining_weits[:, 0]

	# put the conversions together and return them as functions
	def reduced(full):
		return full[important, :]
	def restored(reduced):
		return (reduced[defining_indices, :]*defining_weits[:, :, np.newaxis]).sum(axis=1)
	return reduced, restored


def compute_principal_strains(??: np.ndarray, cell_definitions: np.ndarray,
                              positions: np.ndarray) -> tuple[float, float]:
	""" take a set of cell definitions and 2D coordinates for each node, and calculate
	    the Tissot-ellipse semiaxes of each cell.
	"""
	i = cell_definitions[:, 1]
	d?? = EARTH.R*d??*np.cos(??[i]) # TODO: account for eccentricity
	d?? = EARTH.R*d??

	west = positions[cell_definitions[:, 3], :]
	east = positions[cell_definitions[:, 4], :]
	dxd?? = ((east - west)/d??[:, None])[:, 0]
	dyd?? = ((east - west)/d??[:, None])[:, 1]

	south = positions[cell_definitions[:, 5], :]
	north = positions[cell_definitions[:, 6], :]
	dxd?? = ((north - south)/d??)[:, 0]
	dyd?? = ((north - south)/d??)[:, 1]

	trace = np.sqrt((dxd?? + dyd??)**2 + (dxd?? - dyd??)**2)/2
	antitrace = np.sqrt((dxd?? - dyd??)**2 + (dxd?? + dyd??)**2)/2
	return trace + antitrace, trace - antitrace


def load_pixel_values(filename: str) -> np.ndarray:
	""" load and resample a generic 2D raster image """
	if filename == "uniform":
		return np.array(1)
	else:
		return tifffile.imread(f"../spec/pixels_{filename}.tif")


def load_options(filename: str) -> dict[str, str]:
	""" load a simple colon-separated text file """
	options = dict()
	with open(f"../spec/options_{filename}.txt", "r", encoding="utf-8") as file:
		for line in file.readlines():
			key, value = line.split(":")
			options[key.strip()] = value.strip()
	return options


def load_coastline_data(reduction=2) -> list[np.ndarray]:
	coastlines = []
	with shapefile.Reader(f"../data/ne_110m_coastline.zip") as shapef:
		for shape in shapef.shapes():
			if len(shape.points) > 3*reduction:
				coastlines.append(np.radians(shape.points)[::reduction, ::-1])
	return coastlines


def load_mesh(filename: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:
	""" load the ?? values, ?? values, node locations, and section borders from a HDF5
	    file, in that order.
	"""
	with h5py.File(f"../spec/mesh_{filename}.h5", "r") as file: # TODO: I should look into reducing the precision on some of these numbers
		?? = np.radians(file["section0/latitude"])
		?? = np.radians(file["section0/longitude"])
		num_sections = file.attrs["num_sections"]
		mesh = np.empty((num_sections, ??.size, ??.size, 2))
		sections = []
		for h in range(file.attrs["num_sections"]):
			mesh[h, :, :, :] = file[f"section{h}/projection"]
			sections.append(np.radians(file[f"section{h}/border"][:, :]))
	return ??, ??, mesh, sections


def show_mesh(fit_positions: np.ndarray, all_positions: np.ndarray,
              velocity: np.ndarray, values: list[float],
              final: bool, ??_mesh: np.ndarray, ??_mesh: np.ndarray,
              mesh_index: np.ndarray, coastlines: list[np.array],
              map_axes: plt.Axes, hist_axes: plt.Axes,
              valu_axes: plt.Axes, diff_axes: plt.Axes) -> None:
	map_axes.clear()
	mesh = all_positions[mesh_index, :]
	for h in range(mesh.shape[0]):
		map_axes.plot(mesh[h, :, :, 0], mesh[h, :, :, 1], f"#bbb", linewidth=.4) # TODO: zoom in and stuff?
		map_axes.plot(mesh[h, :, :, 0].T, mesh[h, :, :, 1].T, f"#bbb", linewidth=.4)
		project = interpolate.RegularGridInterpolator([??_mesh, ??_mesh], mesh[h, :, :, :],
		                                              bounds_error=False, fill_value=np.nan)
		for line in coastlines:
			projected_line = project(line)
			plt.plot(projected_line[:, 0], projected_line[:, 1], f"#000", linewidth=.8, zorder=2)
	if not final:
		map_axes.scatter(fit_positions[:, 0], fit_positions[:, 1], s=10,
		                 c=-np.linalg.norm(velocity, axis=1),
		                 cmap=CUSTOM_CMAP["speed"]) # TODO: zoom in and rotate automatically
	map_axes.axis("equal")

	a, b = compute_principal_strains(??_mesh, cell_definitions, all_positions)
	hist_axes.clear()
	hist_axes.hist2d(np.concatenate([a, b]),
	                 np.concatenate([b, a]),
	                 weights=np.tile(cell_weights, 2),
	                 bins=np.linspace(0, 2, 41),
	                 cmap=CUSTOM_CMAP["density"])
	hist_axes.axis("square")

	valu_axes.clear()
	valu_axes.plot(values)
	valu_axes.set_xlim(len(values) - 1000, len(values))
	valu_axes.set_ylim(0, 3*values[-1])
	valu_axes.minorticks_on()
	valu_axes.yaxis.set_tick_params(which='both')
	valu_axes.grid(which="both", axis="y")

	diff_axes.clear()
	diffs = -np.diff(values)/values[1:]
	if diffs.size > 0:
		diff_axes.scatter(np.arange(1, len(values)), diffs, s=1, zorder=10)
	diff_axes.scatter(np.arange(len(grads)), grads, s=1, zorder=10)
	ylim = max(2e-2, diffs.min(where=diffs != 0, initial=np.min(grads))*1e3)
	diff_axes.set_ylim(ylim/1e3, ylim)
	diff_axes.set_yscale("log")
	diff_axes.grid(which="major", axis="y")


def save_mesh(name: str, ??: np.ndarray, ??: np.ndarray, mesh: np.ndarray,
              section_borders: list[np.ndarray], section_names: list[str],
              descript: str) -> None:
	""" save all of the important map projection information as a HDF5 file.
	    :param name: the name of this elastick map projection
	    :param ??: the m latitude values at which the projection is defined
	    :param ??: the l longitude values at which the projection is defined
	    :param mesh: an n??m??l??2 array of the x and y coordinates at each point
	    :param section_borders: a list of the borders of the n sections. each one is an
	                            o??2 array of x and y coordinates, starting and ending at
	                            the same point.
	    :param section_names: a list of the names of the n sections. these will be added
	                          to the HDF5 file as attributes.
	    :param descript: a short description of the map projection, to be included in the
	                     HDF5 file as an attribute.
	"""
	assert len(section_borders) == len(section_names)

	with h5py.File(f"../projection/elastik-{name}.h5", "w") as file:
		file.attrs["name"] = name
		file.attrs["description"] = descript
		file.attrs["num_sections"] = len(section_borders)
		dset = file.create_dataset("bounding_box", shape=(2, 2))
		dset.attrs["units"] = "km"
		dset[:, :] = get_bounding_box(mesh)
		dset = file.create_dataset("sections", shape=(len(section_borders),), dtype=h5_str)
		dset[:] = [f"section{i}" for i in range(len(section_borders))]

		for h in range(len(section_borders)):
			i_relevant = dilate(np.any(~np.isnan(mesh[h, :, :, 0]), axis=1), 1)
			num_?? = np.sum(i_relevant)
			j_relevant = dilate(np.any(~np.isnan(mesh[h, 1:-1, :, 0]), axis=0), 1)
			num_?? = np.sum(j_relevant)

			group = file.create_group(f"section{h}")
			group.attrs["name"] = section_names[h]
			dset = group.create_dataset("latitude", shape=(num_??,))
			dset.attrs["units"] = "??"
			dset[:] = np.degrees(??[i_relevant])
			dset = group.create_dataset("longitude", shape=(num_??,))
			dset.attrs["units"] = "??"
			dset[:] = np.degrees(??[j_relevant])
			dset = group.create_dataset("projection", shape=(num_??, num_??, 2))
			dset.attrs["units"] = "km"
			dset[:, :, :] = mesh[h][i_relevant][:, j_relevant]
			dset = group.create_dataset("border", shape=section_borders[h].shape)
			dset.attrs["units"] = "??"
			dset[:, :] = np.degrees(section_borders[h])
			dset = group.create_dataset("bounding_box", shape=(2, 2))
			dset.attrs["units"] = "km"
			dset[:, :] = get_bounding_box(mesh[h, :, :, :])


if __name__ == "__main__":
	configure = load_options(CONFIGURATION_FILE)
	??_mesh, ??_mesh, mesh, section_borders = load_mesh(configure["cuts"])
	scale = downsample(load_pixel_values(configure["scale"]), mesh.shape[1:3]) # I get best results when values is
	values = downsample(load_pixel_values(configure["weights"])**2, mesh.shape[1:3]) # steeper than scale, hence this ^2

	# assume the coordinates are more or less evenly spaced
	d?? = ??_mesh[1] - ??_mesh[0]
	d?? = ??_mesh[1] - ??_mesh[0]

	# reformat the nodes into a list without gaps or duplicates
	node_indices, initial_node_positions = enumerate_nodes(mesh)

	# and then do the same thing for cell corners
	cell_definitions, cell_weights = enumerate_cells(??_mesh, node_indices, values)

	# define functions that can define the node positions from a reduced set of them
	reduce, restore = mesh_skeleton(??_mesh, node_indices)

	# load the coastline data from Natural Earth
	coastlines = load_coastline_data()

	# set up the plotting axes
	small_fig = plt.figure(figsize=(3, 5))
	gridspecs = (plt.GridSpec(3, 1, height_ratios=[2, 1, 1]),
	             plt.GridSpec(3, 1, height_ratios=[2, 1, 1], hspace=0))
	hist_axes = small_fig.add_subplot(gridspecs[0][0, :])
	valu_axes = small_fig.add_subplot(gridspecs[1][1, :])
	diff_axes = small_fig.add_subplot(gridspecs[1][2, :], sharex=valu_axes)
	main_fig, map_axes = plt.subplots(figsize=(7, 5))

	values, grads = [], []

	# define the objective functions
	def compute_energy_lenient(positions: np.ndarray) -> float:
		a, b = compute_principal_strains(??_mesh, cell_definitions, restore(positions))
		scale_term = (a*b - 1)**2
		shape_term = (a - b)**2
		return ((scale_term + 3*shape_term)*cell_weights).sum()

	def compute_energy_strict(positions: np.ndarray) -> float:
		a, b = compute_principal_strains(??_mesh, cell_definitions, positions)
		if np.any(a <= 0) or np.any(b <= 0):
			return np.inf
		ab = a*b
		scale_term = (ab**2 - 1)/2 - GradientSafe.log(ab)
		shape_term = (a - b)**2
		return ((scale_term + 3*shape_term)*cell_weights).sum()

	def plot_status(positions: np.ndarray, value: float, grad: np.ndarray, step: np.ndarray, final: bool) -> None:
		values.append(value)
		grads.append(np.linalg.norm(grad)*EARTH.R)
		if np.random.random() < 1e-1 or final:
			if positions.shape[0] == initial_node_positions.shape[0]:
				all_positions = np.concatenate([positions, [[np.nan, np.nan]]])
			else:
				all_positions = np.concatenate([restore(positions), [[np.nan, np.nan]]])
			show_mesh(positions, all_positions, step, values, final,
			          ??_mesh, ??_mesh, node_indices, coastlines,
			          map_axes, hist_axes, valu_axes, diff_axes)
			main_fig.canvas.draw()
			small_fig.canvas.draw()
			plt.pause(.01)

	# then minimize!
	print("begin fitting process")
	# start with the lenient condition, since the initial gess is likely to have inside-out cells
	node_positions = minimize(compute_energy_lenient,
	                          guess=reduce(initial_node_positions),
	                          bounds=None,
	                          report=plot_status,
	                          tolerance=1e-3/EARTH.R)

	# this should make the mesh well-behaved
	assert np.all(np.positive(compute_principal_strains(
		??_mesh, cell_definitions, restore(node_positions))))

	# then switch to the strict condition and full mesh
	node_positions = minimize(compute_energy_strict,
	                          guess=restore(node_positions),
	                          bounds=None,
	                          report=plot_status,
	                          tolerance=1e-3/EARTH.R)

	# apply the optimized vector back to the mesh
	mesh = node_positions[node_indices, :]
	mesh[node_indices == -1, :] = np.nan

	# and save it!
	save_mesh(configure["name"], ??_mesh, ??_mesh, mesh,
	          section_borders, configure["section_names"].split(","),
	          configure["descript"])

	plt.show()
