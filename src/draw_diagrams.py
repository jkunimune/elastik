"""
draw_diagrams.py

generate some explanatory images to help readers understand how
to use these map projections
"""
from math import nan

import h5py
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from scipy.interpolate import RegularGridInterpolator

from build_mesh import build_mesh
from create_map_projection import create_map_projection, Mesh, load_coastline_data


def draw_diagrams():
	plt.rcParams.update({'font.size': 12})

	elastic_earth = load_mesh("elastic-earth-I")
	rectangular = equirectangular_like(elastic_earth)

	fig, (ax_left, ax_right) = plt.subplots(1, 2, num="all sections", figsize=(7, 4))
	ax_left.set_xlabel("Longitude (°)")
	ax_left.set_ylabel("Latitude (°)")
	draw_section(ax_left, rectangular, 0, nodes=True, edges=True, border=False, graticule=True, coastlines=True)
	ax_right.set_xlabel("x (km)")
	ax_right.set_ylabel("y (km)", rotation=-90)
	draw_section(ax_right, elastic_earth, 0, nodes=True, edges=True, border=False, graticule=True, coastlines=True)
	ax_right.yaxis.tick_right()
	ax_right.yaxis.set_label_position("right")
	# TODO: manually set ticks
	plt.tight_layout()
	plt.savefig("../examples/explanation-1.png", dpi=150)

	plt.show()


def draw_section(ax: Axes, mesh: Mesh, h: int,
                 nodes: bool, edges: bool, border: bool,
                 graticule: bool, coastlines: bool) -> None:
	if nodes:
		ax.scatter(mesh.nodes[h, :, :, 0], mesh.nodes[h, :, :, 1], color="k", s=25, zorder=20)
	if edges:
		ax.plot(mesh.nodes[h, :, :, 0], mesh.nodes[h, :, :, 1], color="k", linewidth=1.5, zorder=20)
		ax.plot(mesh.nodes[h, :, :, 0].T, mesh.nodes[h, :, :, 1].T, color="k", linewidth=1.5, zorder=20)
	if coastlines:
		project = RegularGridInterpolator([mesh.ф, mesh.λ], mesh.nodes[h, :, :, :],
		                                  bounds_error=False, fill_value=nan)
		coastlines = load_coastline_data(reduction=1)
		for line in coastlines:
			projected_line = project(line)
			ax.plot(projected_line[:, 0], projected_line[:, 1], "#596d74", linewidth=0.8, zorder=10)
	ax.axis("equal")


def load_mesh(filename) -> Mesh:
	with h5py.File(f"../projection/{filename}.h5", "r") as file:
		ф = file["section 0/projected points/latitude"][:]
		λ = file["section 0/projected points/longitude"][:]
		num_sections = file["sections"].size
		nodes = np.empty((num_sections, ф.size, λ.size, 2))
		section_borders = []
		for h, section in enumerate(file["sections"]):
			section = section.decode()
			nodes[h, :, :, 0] = file[f"{section}/projected points/points"][:, :]["x"]
			nodes[h, :, :, 1] = file[f"{section}/projected points/points"][:, :]["y"]
			section_borders.append(file[f"{section}/border"][:])
	return Mesh(section_borders, ф, λ, nodes)


def equirectangular_like(mesh: Mesh) -> Mesh:
	Φ, Λ = np.meshgrid(mesh.ф, mesh.λ, indexing="ij")
	nodes = np.empty_like(mesh.nodes)
	nodes[:, :, :, 0] = Λ
	nodes[:, :, :, 1] = Φ
	nodes[np.isnan(mesh.nodes)] = nan
	return Mesh(mesh.section_borders, mesh.ф, mesh.λ, nodes)



if __name__ == "__main__":
	# first compute the Elastic Earth I projection with lower resolution
	if load_mesh("elastic-earth-I").nodes.shape[1] > 10:
		build_mesh("oceans", resolution=4)
		create_map_projection("continents")
		plt.close("all")
	# then draw the diagrams using that new coarse projection
	draw_diagrams()
