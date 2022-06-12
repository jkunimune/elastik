#!/usr/bin/env python
"""
create_map_projection.py

take a basic mesh and optimize it according to a particular cost function in order to
create a new Elastic Projection.
"""
import h5py
import numpy as np

from optimize import minimize
from util import dilate

CONFIGURATION_FILE = "oceans" # "continents"; "countries"


def find_or_add(vector, vectors) -> int:
	""" add vector to vectors if it's not already there, then return its index """
	if np.any(np.isnan(vector)):
		return -1
	else:
		for k in range(len(vectors)):
			if np.all(vectors[k] == vector):
				return k
		vectors.append(vector)
		return len(vectors) - 1


def load_pixel_values(filename: str) -> np.ndarray:
	""" load a generic 2D raster image """
	if filename == "uniform":
		return np.array(1)
	else:
		return np.loadtxt(f"../spec/pixels_{filename}.txt")


def load_options(filename: str) -> dict[str, str]:
	""" load a simple colon-separated text file """
	options = dict()
	with open(f"../spec/options_{filename}.txt", "r") as file:
		for line in file.readlines():
			key, value = line.split(":")
			options[key.strip()] = value.strip()
	return options


def load_mesh(filename: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:
	""" load the ф values, λ values, node locations, and section borders from a HDF5
	    file, in that order.
	"""
	print(f"../spec/mesh_{filename}.h5")
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
		for h in range(len(section_borders)):
			group = file.create_group(f"section{h}")
			group.attrs["name"] = section_names[h]
			i_relevant = dilate(np.any(~np.isnan(mesh[h, :, :, 0]), axis=1), 1)
			num_ф = np.sum(i_relevant)
			j_relevant = dilate(np.any(~np.isnan(mesh[h, 1:-1, :, 0]), axis=0), 1)
			num_λ = np.sum(j_relevant)
			dset = group.create_dataset("latitude", shape=(num_ф,))
			dset[:] = np.degrees(ф[i_relevant])
			dset = group.create_dataset("longitude", shape=(num_λ,))
			dset[:] = np.degrees(λ[j_relevant])
			dset = group.create_dataset("projection", shape=(num_ф, num_λ, 2))
			dset[:, :, :] = mesh[h][i_relevant][:, j_relevant]
			dset = group.create_dataset("border", shape=section_borders[h].shape)
			dset[:, :] = np.degrees(section_borders[h])


if __name__ == "__main__":
	configure = load_options(CONFIGURATION_FILE)
	ф, λ, mesh, section_borders = load_mesh(configure["cuts"])
	weights = load_pixel_values(configure["weights"])
	scale = load_pixel_values(configure["scale"])

	node_indices = np.empty((len(section_borders), ф.size, λ.size), dtype=int)
	node_positions = []
	for h in range(len(section_borders)):
		for i in range(ф.size):
			for j in range(λ.size):
				node_indices[h, i, j] = find_or_add(mesh[h, i, j, :], node_positions)

	def compute_energy(positions: np.ndarray, arrangement=node_indices) -> float:
		return 0

	def compute_gradient(positions: np.ndarray, arrangement=node_indices) -> np.ndarray:
		return np.zeros_like(positions)

	def plot_status(positions: np.ndarray) -> None:
		pass

	node_positions = minimize(compute_energy,
	                          compute_gradient,
	                          np.array(node_positions),
	                          scale=None,
	                          bounds=None,
	                          report=plot_status)

	save_mesh(configure["name"], ф, λ, mesh,
	          section_borders, configure["section_names"].split(","),
	          configure["descript"])
