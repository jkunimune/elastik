#!/usr/bin/env python
"""
create_map_projection.py

take a basic mesh and optimize it according to a particular cost function in order to
create a new Elastic Projection.
"""
import h5py
import numpy as np
from matplotlib import pyplot as plt

from optimize import minimize, log
from util import dilate, h5_str, EARTH

CONFIGURATION_FILE = "oceans" # "continents"; "countries"


def find_or_add(vector: np.ndarray, vectors: list[np.ndarray]) -> int:
	""" add vector to vectors if it's not already there, then return its index """
	if np.any(np.isnan(vector)):
		return -1
	else:
		for k in range(len(vectors)):
			if np.all(vectors[k] == vector):
				return k
		vectors.append(vector)
		return len(vectors) - 1


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
	node_indices = np.empty((len(section_borders), ф_mesh.size, λ_mesh.size),
	                        dtype=int)
	node_positions = []
	for h in range(len(section_borders)):
		for i in range(ф_mesh.size):
			for j in range(λ_mesh.size):
				if not np.any(np.isnan(mesh[h, i, j, :])):
					node_indices[h, i, j] = find_or_add(mesh[h, i, j, :], node_positions)
				else:
					node_indices[h, i, j] = -1

	# and then do the same thing for cells, defining each by the indices of its nodes
	cell_indices = np.empty((len(section_borders), ф_mesh.size - 1, λ_mesh.size - 1, 2, 2),
	                        dtype=int)
	cell_definitions = []
	for h in range(len(section_borders)):
		for i in range(ф_mesh.size - 1):
			for j in range(λ_mesh.size - 1):
				for di in range(0, 2): # each corner of each cell should be represented separately
					for dj in range(0, 2):
						i_primary = i + di
						i_secondary = i + 1 - di
						if i_primary != 0 and i_primary != ф_mesh.size - 1: # skip the corners next to the poles
							west_node = node_indices[h, i + di, j]
							east_node = node_indices[h, i + di, j + 1]
							south_node = node_indices[h, i,     j + dj]
							north_node = node_indices[h, i + 1, j + dj]
							cell_definition = np.array([i_primary, i_secondary,
							                            west_node, east_node,
							                            south_node, north_node])
							if not np.any(cell_definition == -1):
								cell_indices[h, i, j, di, dj] = find_or_add(
									cell_definition, cell_definitions)
							else:
								cell_indices[h, i, j, di, dj] = -1

	cell_areas = []
	for k in range(len(cell_definitions)):
		i_prim, i_second = cell_definitions[k][:2]
		ф_prim, ф_second = ф_mesh[i_prim], ф_mesh[i_second]
		cell_areas.append(EARTH.R**2*dф*dλ*(3*np.cos(ф_prim) + np.cos(ф_second))/16)

	# now we're all set to do this calculation in numpy
	node_positions = np.array(node_positions)
	cell_definitions = np.array(cell_definitions)
	cell_areas = np.array(cell_areas)

	def compute_principal_strains(positions: np.ndarray) -> tuple[float, float]:
		i = cell_definitions[:, 0]
		dΛ = EARTH.R*dλ*np.cos(ф_mesh[i]) # TODO: account for eccentricity
		dΦ = EARTH.R*dф

		west = positions[cell_definitions[:, 2], :]
		east = positions[cell_definitions[:, 3], :]
		dxdΛ = ((east - west)/dΛ[:, None])[:, 0]
		dydΛ = ((east - west)/dΛ[:, None])[:, 1]

		south = positions[cell_definitions[:, 4], :]
		north = positions[cell_definitions[:, 5], :]
		dxdΦ = ((north - south)/dΦ)[:, 0]
		dydΦ = ((north - south)/dΦ)[:, 1]

		trace = np.sqrt((dxdΛ + dydΦ)**2 + (dxdΦ - dydΛ)**2)
		antitrace = np.sqrt((dxdΛ - dydΦ)**2 + (dxdΦ + dydΛ)**2)
		return trace + antitrace, trace - antitrace

	def compute_energy_loose(positions: np.ndarray) -> float:
		a, b = compute_principal_strains(positions)
		scale_term = (a*b - 1)**2
		shape_term = (a - b)**2
		return ((3*scale_term + shape_term)*cell_areas).sum()

	def compute_energy_strict(positions: np.ndarray, arrangement=node_indices) -> float:
		a, b = compute_principal_strains(positions)
		if np.any(a <= 0) or np.any(b <= 0):
			print(a, b)
			raise ValueError("the principal stretches should all be positive by now")
		ab = a*b
		scale_term = ab**2/2 - log(ab)
		shape_term = ((a/b + b/a)/2)**2
		return ((3*scale_term + shape_term)*cell_areas).sum()

	def plot_status(positions: np.ndarray, value: float, step: np.ndarray) -> None:
		print(value)
		if np.random.random() < 3e-1:
			plt.clf()
			plt.scatter(positions[:, 0], positions[:, 1], c=-np.hypot(step[:, 0], step[:, 1])) # TODO: zoom in and rotate automatically
			plt.axis("equal")
			plt.pause(.01)

	node_positions = minimize(compute_energy_loose,
	                          guess=node_positions,
	                          bounds=None,
	                          report=plot_status) # TODO: scale gradients

	node_positions = minimize(compute_energy_strict,
	                          guess=node_positions,
	                          bounds=None,
	                          report=plot_status)

	mesh = node_positions[node_indices, :]
	mesh[node_indices == -1, :] = np.nan

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
