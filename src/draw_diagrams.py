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
from matplotlib.ticker import MultipleLocator
from scipy.interpolate import RegularGridInterpolator

from build_mesh import build_mesh
from create_map_projection import create_map_projection, Mesh, load_coastline_data
from util import refine_path


def draw_diagrams():
	plt.rcParams.update({'font.size': 12})

	# first compute the Elastic Earth I projection with lower resolution
	if load_mesh("elastic-earth-I").nodes.shape[1] > 10:
		build_mesh("oceans", resolution=4)
		create_map_projection("continents")
		plt.close("all")

	# then draw the diagrams using that new coarse projection
	mesh = load_mesh("elastic-earth-I")
	mesh.nodes /= 1e3  # scale down to a realistic map size and change km to cm
	section_index = 0

	fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(7, 4))
	plot_projection_domains(
		ax_left, ax_right, mesh, section_index, color="k",
		nodes=True, boundary=False, shading=False, graticule=False, coastlines=False)
	plt.savefig("../resources/diagram-1.png", dpi=150)

	fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(7, 4))
	plot_projection_domains(
		ax_left, ax_right, mesh, section_index, color="k",
		nodes=True, boundary=False, shading=True, graticule=True, coastlines=True)
	plt.savefig("../resources/diagram-2.png", dpi=150)

	fig, ax = plt.subplots(1, 1, figsize=(7, 4))
	for index, color in enumerate(["#001D47", "#0C2C03", "#5F1021"]):
		draw_section(ax, mesh, index, color,
		             nodes=True, boundary=False, shading=True, graticule=True, coastlines=True)
	ax_right.set_xlabel("x (at 1:100M scale)")
	ax_right.set_ylabel("y (at 1:100M scale)", labelpad=11, rotation=-90)
	set_ticks(ax_right, spacing=5, fmt="{x:.0f} cm", y_ticks_on_right=True)
	plt.savefig("../resources/diagram-3.png", dpi=150)

	fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(7, 4))
	plot_projection_domains(
		ax_left, ax_right, mesh, section_index, color="k",
		nodes=True, boundary=True, shading=True, graticule=True, coastlines=True)
	plt.savefig("../resources/diagram-4.png", dpi=150)

	fig, ax = plt.subplots(1, 1, figsize=(7, 4))
	for index, color in enumerate(["#001D47", "#0C2C03", "#5F1021"]):
		draw_section(ax, mesh, index, color,
		             nodes=True, boundary=True, shading=True, graticule=True, coastlines=True)
	ax_right.set_xlabel("x (at 1:100M scale)")
	ax_right.set_ylabel("y (at 1:100M scale)", labelpad=11, rotation=-90)
	set_ticks(ax_right, spacing=5, fmt="{x:.0f} cm", y_ticks_on_right=True)
	plt.savefig("../resources/diagram-5.png", dpi=150)

	plt.show()


def plot_projection_domains(ax_left: Axes, ax_right: Axes,
                            elastic_earth_mesh: Mesh, section_index: int, color: str,
                            nodes: bool, boundary: bool, shading: bool,
                            graticule: bool, coastlines: bool) -> None:
	equirectangular_mesh = equirectangular_like(elastic_earth_mesh)
	draw_section(ax_left, equirectangular_mesh, section_index, color,
	             nodes, boundary, shading, graticule, coastlines)
	ax_left.set_xlabel("Longitude")
	ax_left.set_ylabel("Latitude", labelpad=-1)
	set_ticks(ax_left, spacing=30, fmt="{x:.0f}°")
	draw_section(ax_right, elastic_earth_mesh, section_index, color,
	             nodes, boundary, shading, graticule, coastlines)
	ax_right.set_xlabel("x (at 1:100M scale)")
	ax_right.set_ylabel("y (at 1:100M scale)", labelpad=11, rotation=-90)
	set_ticks(ax_right, spacing=5, fmt="{x:.0f} cm", y_ticks_on_right=True)
	plt.tight_layout()



def draw_section(ax: Axes, mesh: Mesh, section_index: int, color: str,
                 nodes: bool, boundary: bool, shading: bool,
                 graticule: bool, coastlines: bool) -> None:
	if nodes:
		ax.scatter(mesh.nodes[section_index, :, :, 0], mesh.nodes[section_index, :, :, 1],
		           color=color, s=10, zorder=10)
	if graticule:
		for nodes in [mesh.nodes[section_index], mesh.nodes[section_index].transpose((1, 0, 2))]:
			for weit in np.linspace(0, 1, 3, endpoint=False):
				j = np.arange(nodes.shape[1])
				if weit != 0:
					weited_nodes = weit*nodes[:, j - 1, :] + (1 - weit)*nodes[:, j, :]
				else:
					weited_nodes = nodes[:, j, :]
				ax.plot(weited_nodes[:, :, 0], weited_nodes[:, :, 1],
				        color=color, linewidth=0.5, zorder=10)
	if coastlines:
		project = RegularGridInterpolator([mesh.ф, mesh.λ], mesh.nodes[section_index, :, :, :],
		                                  bounds_error=False, fill_value=nan)
		coastlines = load_coastline_data(reduction=1)
		for line in coastlines:
			projected_line = project(line)
			ax.plot(projected_line[:, 0], projected_line[:, 1],
			        color, linewidth=1.0, zorder=20)
	if boundary:
		project = RegularGridInterpolator([mesh.ф, mesh.λ], mesh.nodes[section_index, :, :, :],
		                                  bounds_error=False, fill_value=nan)
		projected_boundary = project(refine_path(
			mesh.section_boundaries[section_index], resolution=1))
		ax.plot(projected_boundary[:, 0], projected_boundary[:, 1],
		        color, linewidth=2.0, zorder=30)
	ax.axis("equal")


def set_ticks(ax: Axes, spacing: float, fmt: str, y_ticks_on_right=False) -> None:
	for axis in [ax.xaxis, ax.yaxis]:
		axis.set_major_locator(MultipleLocator(spacing))
		axis.set_major_formatter(fmt)
	if y_ticks_on_right:
		ax.yaxis.tick_right()
		ax.yaxis.set_label_position("right")


def load_mesh(filename) -> Mesh:
	with h5py.File(f"../projection/{filename}.h5", "r") as file:
		ф = file["section 0/projected points/latitude"][:]
		λ = file["section 0/projected points/longitude"][:]
		num_sections = file["sections"].size
		nodes = np.empty((num_sections, ф.size, λ.size, 2))
		section_boundaries = []
		for h, section in enumerate(file["sections"]):
			section = section.decode()
			nodes[h, :, :, 0] = file[f"{section}/projected points/points"][:, :]["x"]
			nodes[h, :, :, 1] = file[f"{section}/projected points/points"][:, :]["y"]
			boundary = file[f"{section}/boundary"][:]
			boundary = np.stack([boundary["latitude"], boundary["longitude"]], axis=-1)
			section_boundaries.append(boundary)
	return Mesh(section_boundaries, ф, λ, nodes)


def equirectangular_like(mesh: Mesh) -> Mesh:
	Φ, Λ = np.meshgrid(mesh.ф, mesh.λ, indexing="ij")
	nodes = np.empty_like(mesh.nodes)
	nodes[:, :, :, 0] = Λ
	nodes[:, :, :, 1] = Φ
	nodes[np.isnan(mesh.nodes)] = nan
	return Mesh(mesh.section_boundaries, mesh.ф, mesh.λ, nodes)



if __name__ == "__main__":
	draw_diagrams()
