#!/usr/bin/env python
"""
create_map_projection.py

take a basic mesh and optimize it according to a particular cost function in order to
create a new Elastic Projection.
"""
from typing import Callable

import h5py
import numpy as np
from matplotlib import pyplot as plt

from optimize import minimize, GradientSafe
from util import dilate, h5_str, EARTH

CONFIGURATION_FILE = "oceans" # "continents"; "countries"


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
	    returning mappings from the old mesh to the new indices and the n×2 list positions
	"""
	node_indices = np.empty(mesh.shape[:3], dtype=int)
	node_positions = []
	for h in range(mesh.shape[0]):
		for i in range(mesh.shape[1]):
			for j in range(mesh.shape[2]):
				_, node_indices[h, i, j] = find_or_add(mesh[h, i, j, :], node_positions)
	return node_indices, np.array(node_positions)


def enumerate_cells(ф: np.ndarray, node_indices: np.ndarray
                    ) -> tuple[np.ndarray, np.ndarray]:
	""" take an array of nodes and generate the list of cells in which the elastic energy
	    should be calculated.
	"""
	cell_definitions = []
	cell_areas = []
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
									ф_1 = ф[i + di]
									ф_2 = ф[i + 1 - di]
									cell_areas.append(EARTH.R**2*dф*dλ*(3*np.cos(ф_1) + np.cos(ф_2))/16)
	return np.array(cell_definitions), np.array(cell_areas)/(4*np.pi*EARTH.R**2)


def mesh_skeleton(ф: np.ndarray, lookup_table: np.ndarray
                  ) -> tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]:
	""" create a pair of inverse functions that transform points between the full space of
	    possible meshes and a reduced space with fewer degrees of freedom. the idea here
	    is to identify 80% or so of the nodes that can form a skeleton, and from which the
	    remaining nodes can be interpolated. to that end, each parallel will have some
	    number of regularly spaced key nodes, and all nodes adjacent to an edge will
	    be keys, as well, and all nodes that aren't key nodes will be ignored when
	    reducing the position vector and interpolated from neibors when restoring it.
	    :param ф: the latitudes of the nodes
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
			period = int(round(1/np.cos(ф[i])))
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


def get_bounding_box(points: np.ndarray) -> np.ndarray:
	""" compute the maximum and minimums of this set of points and package it as
	    [[left, bottom], [right, top]]
	"""
	return np.array([
		[np.nanmin(points[..., 0]), np.nanmin(points[..., 1])],
		[np.nanmax(points[..., 0]), np.nanmax(points[..., 1])], # TODO: account for the border, and for spline interpolation
	])


def load_pixel_values(filename: str) -> np.ndarray:
	""" load a generic 2D raster image """
	if filename == "uniform":
		return np.array(1)
	else:
		return np.loadtxt(f"../spec/pixels_{filename}.txt")


def load_options(filename: str) -> dict[str, str]:
	""" load a simple colon-separated text file """
	options = dict()
	with open(f"../spec/options_{filename}.txt", "r", encoding="utf-8") as file:
		for line in file.readlines():
			key, value = line.split(":")
			options[key.strip()] = value.strip()
	return options


def load_mesh(filename: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:
	""" load the ф values, λ values, node locations, and section borders from a HDF5
	    file, in that order.
	"""
	with h5py.File(f"../spec/mesh_{filename}.h5", "r") as file: # TODO: I should look into reducing the precision on some of these numbers
		ф = np.radians(file["section0/latitude"])
		λ = np.radians(file["section0/longitude"])
		num_sections = file.attrs["num_sections"]
		mesh = np.empty((num_sections, ф.size, λ.size, 2))
		sections = []
		for h in range(file.attrs["num_sections"]):
			mesh[h, :, :, :] = file[f"section{h}/projection"]
			sections.append(np.radians(file[f"section{h}/border"][:, :]))
	return ф, λ, mesh, sections


def save_mesh(name: str, ф: np.ndarray, λ: np.ndarray, mesh: np.ndarray,
              section_borders: list[np.ndarray], section_names: list[str],
              descript: str) -> None:
	""" save all of the important map projection information as a HDF5 file.
	    :param name: the name of this elastick map projection
	    :param ф: the m latitude values at which the projection is defined
	    :param λ: the l longitude values at which the projection is defined
	    :param mesh: an n×m×l×2 array of the x and y coordinates at each point
	    :param section_borders: a list of the borders of the n sections. each one is an
	                            o×2 array of x and y coordinates, starting and ending at
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
			num_ф = np.sum(i_relevant)
			j_relevant = dilate(np.any(~np.isnan(mesh[h, 1:-1, :, 0]), axis=0), 1)
			num_λ = np.sum(j_relevant)

			group = file.create_group(f"section{h}")
			group.attrs["name"] = section_names[h]
			dset = group.create_dataset("latitude", shape=(num_ф,))
			dset.attrs["units"] = "°"
			dset[:] = np.degrees(ф[i_relevant])
			dset = group.create_dataset("longitude", shape=(num_λ,))
			dset.attrs["units"] = "°"
			dset[:] = np.degrees(λ[j_relevant])
			dset = group.create_dataset("projection", shape=(num_ф, num_λ, 2))
			dset.attrs["units"] = "km"
			dset[:, :, :] = mesh[h][i_relevant][:, j_relevant]
			dset = group.create_dataset("border", shape=section_borders[h].shape)
			dset.attrs["units"] = "°"
			dset[:, :] = np.degrees(section_borders[h])
			dset = group.create_dataset("bounding_box", shape=(2, 2))
			dset.attrs["units"] = "km"
			dset[:, :] = get_bounding_box(mesh[h, :, :, :])


if __name__ == "__main__":
	configure = load_options(CONFIGURATION_FILE)
	ф_mesh, λ_mesh, mesh, section_borders = load_mesh(configure["cuts"])
	weights = load_pixel_values(configure["weights"])
	scale = load_pixel_values(configure["scale"])

	# assume the coordinates are more or less evenly spaced
	dλ = λ_mesh[1] - λ_mesh[0]
	dф = ф_mesh[1] - ф_mesh[0]

	# reformat the nodes into a list without gaps or duplicates
	node_indices, initial_node_positions = enumerate_nodes(mesh)

	# and then do the same thing for cell corners
	cell_definitions, cell_areas = enumerate_cells(ф_mesh, node_indices)

	# define functions that can define the node positions from a reduced set of them
	reduced, restored = mesh_skeleton(ф_mesh, node_indices)

	# define the objective functions
	def compute_principal_strains(positions: np.ndarray) -> tuple[float, float]:
		if positions.shape[0] != initial_node_positions.shape[0]: # convert from reduced mesh to full mesh
			return compute_principal_strains(restored(positions))
		assert positions.shape == initial_node_positions.shape

		i = cell_definitions[:, 1]
		dΛ = EARTH.R*dλ*np.cos(ф_mesh[i]) # TODO: account for eccentricity
		dΦ = EARTH.R*dф

		west = positions[cell_definitions[:, 3], :]
		east = positions[cell_definitions[:, 4], :]
		dxdΛ = ((east - west)/dΛ[:, None])[:, 0]
		dydΛ = ((east - west)/dΛ[:, None])[:, 1]

		south = positions[cell_definitions[:, 5], :]
		north = positions[cell_definitions[:, 6], :]
		dxdΦ = ((north - south)/dΦ)[:, 0]
		dydΦ = ((north - south)/dΦ)[:, 1]

		trace = np.sqrt((dxdΛ + dydΦ)**2 + (dxdΦ - dydΛ)**2)
		antitrace = np.sqrt((dxdΛ - dydΦ)**2 + (dxdΦ + dydΛ)**2)
		# if not hasattr(east, "gradients"):
		# 	if np.any(antitrace >= trace):
		# 		for i in np.nonzero(antitrace >= trace)[0]:
		# 			print(f"{west[i]} to {east[i]}; {south[i]} to {north[i]}.  the partials are [[{dxdΛ[i]}, {dxdΦ[i]}], [{dydΛ[i]}, {dydΦ[i]}]], and the symmetric eigenvalues are {trace[i] + antitrace[i]} and {trace[i] - antitrace[i]}")
		# 			break
		return trace + antitrace, trace - antitrace

	def compute_energy_lenient(positions: np.ndarray) -> float:
		a, b = compute_principal_strains(positions)
		scale_term = (a*b - 1)**2
		shape_term = (a - b)**2
		return ((scale_term + 3*shape_term)*cell_areas).sum()

	def compute_energy_strict(positions: np.ndarray) -> float:
		a, b = compute_principal_strains(positions)
		if np.any(a <= 0) or np.any(b <= 0):
			return np.inf
		ab = a*b
		scale_term = ab**2/2 - GradientSafe.log(ab)
		shape_term = ((a/b + b/a)/2)**2
		return ((scale_term + 3*shape_term)*cell_areas).sum()

	def plot_status(positions: np.ndarray, value: float, step: np.ndarray) -> None:
		if np.random.random() < 1e-1:
			print(f"{value:.6f}")
			plt.clf()
			plt.scatter(positions[:, 0], positions[:, 1], c=-np.hypot(step[:, 0], step[:, 1]), s=10) # TODO: zoom in and rotate automatically
			plt.axis("equal")
			plt.pause(.01)

	# then minimize! start with the lenient condition, since the initial gess is likely to have inside-out cells
	node_positions = minimize(compute_energy_lenient,
	                          guess=reduced(initial_node_positions),
	                          bounds=None,
	                          report=plot_status) # TODO: scale gradients

	# this should make the mesh well-behaved
	assert np.all(np.positive(compute_principal_strains(node_positions)))

	# switch to the strict condition
	node_positions = minimize(compute_energy_strict,
	                          guess=node_positions,
	                          bounds=None,
	                          report=plot_status)

	# then finally, do a final pass with the full mesh (rather than the reduced set)
	node_positions = minimize(compute_energy_strict,
	                          guess=restored(node_positions),
	                          bounds=None,
	                          report=plot_status)

	# apply the optimized vector back to the mesh
	mesh = node_positions[node_indices, :]
	mesh[node_indices == -1, :] = np.nan

	# and save it!
	save_mesh(configure["name"], ф_mesh, λ_mesh, mesh,
	          section_borders, configure["section_names"].split(","),
	          configure["descript"])

	plt.close("all")
	plt.figure()
	for h in range(mesh.shape[0]):
		plt.plot(mesh[h, :, :, 0], mesh[h, :, :, 1], f"C{h}") # TODO: zoom in and stuff?
		plt.plot(mesh[h, :, :, 0].T, mesh[h, :, :, 1].T, f"C{h}")
		plt.axis("equal")
	plt.show()
