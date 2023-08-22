"""
draw_diagrams.py

generate some explanatory images to help readers understand how
to use these map projections
"""
import os
from math import nan

import h5py
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Polygon
from matplotlib.ticker import MultipleLocator
from scipy.interpolate import RegularGridInterpolator

from build_mesh import build_mesh
from create_map_projection import create_map_projection, Mesh, load_coastline_data
from util import refine_path, find_boundaries


def draw_diagrams():
	""" generate the visual aides used in the explanation of how to use the projections """
	plt.rcParams.update({'font.size': 12})
	os.makedirs("../resources/images", exist_ok=True)

	# first compute the Elastic Earth I projection with lower resolution
	if not os.path.isfile("../projection/elastic-earth-IV.h5"):
		build_mesh("example", resolution=4)
		create_map_projection("example")
		plt.close("all")

	# then draw the diagrams using that new coarse projection
	mesh = load_mesh("elastic-earth-IV")
	mesh.nodes /= 1e3  # scale down to a realistic map size and change km to cm
	section_index = 0

	# figure 1: nodes in two domains
	fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(7.0, 3.8))
	plot_projection_domains(
		ax_left, ax_right, mesh, section_index, color="#000000",
		nodes=True, boundary=False, shading=False, graticule=False, coastlines=False)
	plt.savefig("../resources/images/diagram-1.png", dpi=80)

	# figure 2: interpolation in two domains
	fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(7.0, 3.8))
	plot_projection_domains(
		ax_left, ax_right, mesh, section_index, color="#000000",
		nodes=True, boundary=False, shading=True, graticule=True, coastlines=True)
	plt.savefig("../resources/images/diagram-2.png", dpi=80)

	# figure 3: complete map
	fig, ax = plt.subplots(1, 1, figsize=(6.0, 4.5))
	for index, color in enumerate(["#001D47", "#0C2C03", "#5F1021"]):
		draw_section(ax, mesh, index, color,
		             nodes=True, boundary=False, shading=True, graticule=False, coastlines=True)
	ax_right.set_xlabel("x (at 1:100M scale)")
	ax_right.set_ylabel("y (at 1:100M scale)", labelpad=11, rotation=-90)
	set_ticks(ax_right, spacing=5, fmt="{x:.0f} cm", y_ticks_on_right=True)
	plt.margins(.01)
	plt.axis("off")
	plt.tight_layout()
	plt.savefig("../resources/images/diagram-3.png", dpi=80)

	# figure 4: boundaries in two domains
	fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(7.0, 3.8))
	plot_projection_domains(
		ax_left, ax_right, mesh, section_index, color="#000000",
		nodes=True, boundary=True, shading=True, graticule=False, coastlines=True)
	plt.savefig("../resources/images/diagram-4.png", dpi=80)

	# figure 5: complete map with boundaries
	fig, ax = plt.subplots(1, 1, figsize=(6.0, 4.5))
	for index, color in enumerate(["#001D47", "#0C2C03", "#5F1021"]):
		draw_section(ax, mesh, index, color,
		             nodes=True, boundary=True, shading=True, graticule=False, coastlines=True)
	ax_right.set_xlabel("x (at 1:100M scale)")
	ax_right.set_ylabel("y (at 1:100M scale)", labelpad=11, rotation=-90)
	set_ticks(ax_right, spacing=5, fmt="{x:.0f} cm", y_ticks_on_right=True)
	plt.margins(.01)
	plt.axis("off")
	plt.tight_layout()
	plt.savefig("../resources/images/diagram-5.png", dpi=80)

	plt.show()


def plot_projection_domains(ax_left: Axes, ax_right: Axes,
                            elastic_earth_mesh: Mesh, section_index: int, color: str,
                            nodes: bool, boundary: bool, shading: bool,
                            graticule: bool, coastlines: bool) -> None:
	""" plot some features of a section in both spherical (i.e. equirectangular) and planar
	    (i.e. Elastic Earth) coordinate systems
	"""
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
	""" plot some features of a section on a given coordinate system (i.e. mesh) """
	# project the boundary if desired
	if boundary:
		project = RegularGridInterpolator([mesh.ф, mesh.λ], mesh.nodes[section_index, :, :, :],
		                                  bounds_error=False, fill_value=None)
		projected_boundary = project(refine_path(
			mesh.section_boundaries[section_index], resolution=1))
		projected_boundary = projected_boundary[~np.any(np.isnan(projected_boundary), axis=1), :]  # the boundary can go thru undefined regions, so just cut that stuff out
		boundary_polygon = Polygon(projected_boundary, closed=True)
		boundary_polygons = [boundary_polygon]
	# if you don't project the boundary, calculate the boundary of the mesh region instead
	else:
		boundaries = find_boundaries(
			np.all(np.isfinite(mesh.nodes[section_index]), axis=-1))
		boundary_polygons = []
		for i_boundary, j_boundary in boundaries:
			boundary_polygon = Polygon(mesh.nodes[section_index, i_boundary, j_boundary, :])
			boundary_polygons.append(boundary_polygon)
	# shade in the boundary if desired
	for boundary_polygon in boundary_polygons:
		ax.add_patch(boundary_polygon)
		if boundary:
			boundary_polygon.set_edgecolor("#000000")
			boundary_polygon.set_linewidth(2.0)
		else:
			boundary_polygon.set_edgecolor("none")
		if shading:
			boundary_polygon.set_facecolor(color + "17")
		else:
			boundary_polygon.set_facecolor("none")
		boundary_polygon.set_zorder(40)

	# plot the node positions
	if nodes:
		ax.scatter(mesh.nodes[section_index, :, :, 0], mesh.nodes[section_index, :, :, 1],
		           color=color, s=10, zorder=10)

	# plot a simple graticule (it's easy-ish because we're doing linear interpolation
	if graticule:
		for nodes in [mesh.nodes[section_index], mesh.nodes[section_index].transpose((1, 0, 2))]:
			for weit in np.linspace(0, 1, 3, endpoint=False):
				j = np.arange(nodes.shape[1])
				if weit != 0:
					weited_nodes = weit*nodes[:, j - 1, :] + (1 - weit)*nodes[:, j, :]
				else:
					weited_nodes = nodes[:, j, :]
				lines = ax.plot(weited_nodes[:, :, 0], weited_nodes[:, :, 1],
				                color=color, linewidth=0.5, zorder=10)
				if boundary:
					for line in lines:  # clip the graticule to the boundary
						line.set_clip_path(boundary_polygons[0])

	# project the coastline data if desired
	if coastlines:
		project = RegularGridInterpolator([mesh.ф, mesh.λ], mesh.nodes[section_index, :, :, :],
		                                  bounds_error=False, fill_value=None)
		coastlines = load_coastline_data(reduction=1)
		for coastline in coastlines:
			projected_coastline = project(coastline)
			lines = ax.plot(projected_coastline[:, 0], projected_coastline[:, 1],
			                color, linewidth=1.0, zorder=20)
			if boundary:
				for line in lines:  # clip the map to the boundary
					line.set_clip_path(boundary_polygons[0])

	ax.axis("equal")


def set_ticks(ax: Axes, spacing: float, fmt: str, y_ticks_on_right=False) -> None:
	""" adjust the tick marks of an axes to have a given spacing, number format, and y-axis location """
	for axis in [ax.xaxis, ax.yaxis]:
		axis.set_major_locator(MultipleLocator(spacing))
		axis.set_major_formatter(fmt)
	if y_ticks_on_right:
		ax.yaxis.tick_right()
		ax.yaxis.set_label_position("right")


def load_mesh(filename: str) -> Mesh:
	""" load an Elastic Earth mesh from disk """
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
	""" generate something that looks like an Elastic Earth mesh but corresponds to an
	    equirectangular projection
	"""
	Φ, Λ = np.meshgrid(mesh.ф, mesh.λ, indexing="ij")
	nodes = np.empty_like(mesh.nodes)
	nodes[:, :, :, 0] = Λ
	nodes[:, :, :, 1] = Φ
	nodes[np.isnan(mesh.nodes)] = nan
	return Mesh(mesh.section_boundaries, mesh.ф, mesh.λ, nodes)


if __name__ == "__main__":
	draw_diagrams()
