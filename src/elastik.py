#!/usr/bin/env python
"""
elastik.py

this script is an example of how to use the Elastic projections. enclosed in this file is
everything you need to load the mesh files and project data onto them.
"""
from __future__ import annotations

from math import nan
from typing import Any

import h5py
import numpy as np
import shapefile
from matplotlib import pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from numpy.typing import NDArray
from scipy import interpolate

Style = dict[str, Any]
XYPoint = np.dtype([("x", float), ("y", float)])
XYLine = NDArray[XYPoint]
ΦΛPoint = np.dtype([("latitude", float), ("longitude", float)])
ΦΛLine = NDArray[ΦΛPoint]


def create_map(name: str, projection: str, duplicate: bool, background_style: Style, border_style: Style, data: list[tuple[str, Style]]):
	""" build a new map using an elastik projection and some datasets.
	    :param name: the name with which to save the figure
	    :param projection: the name of the elastik projection to use
	    :param duplicate: whether to plot as much data as you can (rather than showing each feature exactly once)
	    :param background_style: the Style for the otherwise unfilled regions of the map area
	    :param border_style: the Style for the line drawn around the complete map area
	    :param data: a list of elements to include in the map
	"""
	sections, border = load_elastik_projection(projection)

	fig = plt.figure(name)
	ax = fig.subplots()

	ax.fill(border["x"], border["y"], edgecolor="none", **background_style)

	for data_name, style in data:
		data, closed = load_geographic_data(data_name)
		projected_data = project(data, sections)
		if closed:
			points: list[tuple[float, float]] = []
			codes: list[int] = []
			for i, line in enumerate(projected_data):
				for j, point in enumerate(line):
					points.append((point["x"], point["y"]))
					codes.append(Path.MOVETO if j == 0 else Path.LINETO)
				points.append((nan, nan))
				codes.append(Path.CLOSEPOLY)
			path = Path(points, codes)
			patch = PathPatch(path, **style)
			ax.add_patch(patch)
		else:
			for i, line in enumerate(projected_data):
				ax.plot(line["x"], line["y"], **style)

	ax.fill(border["x"], border["y"], facecolor="none", **border_style)

	ax.axis("equal")
	ax.axis("off")
	fig.savefig(f"../examples/{name}.svg",
	            bbox_inches="tight", pad_inches=0)


def project(lines: list[ΦΛLine], projection: list[Section]) -> list[XYLine]:
	""" apply an Elastik projection, defined by a list of sections, to the given series of
	    latitudes and longitudes
	"""
	projected: list[XYLine] = []
	for i, line in enumerate(lines):
		print(f"projecting line {i: 3d}/{len(lines): 3d} ({len(line)} points)")
		projected.append(np.empty(line.size, dtype=XYPoint))
		projected[i][:] = (nan, nan)
		# for each line, project it into whichever section that can accommodate it
		for section in projection:
			in_this_section = section.contains(line)
			projected[i][in_this_section] = section.get_planar_coordinates(line[in_this_section])
		# check that each point was projected by at least one section
		assert not np.any(np.isnan(projected[i]["x"]))
	print(f"projected {len(lines)} lines")
	return projected


def inverse_project(points: np.ndarray, mesh: list[Section]) -> list[ΦΛLine]:
	""" apply the inverse of an Elastik projection, defined by a list of sections, to find
	    the latitudes and longitudes that map to these locations on the map
	"""
	pass  # TODO


def load_elastik_projection(name: str) -> tuple[list[Section], ΦΛLine]:
	""" load the hdf5 file that defines an elastik projection
	    :param name: one of "desa", "hai", or "lin"
	"""
	with h5py.File(f"../projection/elastik-{name}.h5", "r") as file:
		sections = []
		for h in range(file.attrs["num_sections"]):
			sections.append(Section(file[f"section{h}/latitude"][:],
			                        file[f"section{h}/longitude"][:],
			                        file[f"section{h}/projection"][:, :],
			                        file[f"section{h}/border"][:],
			                        ))
		border = file["projected_border"][:]
	return sections, border


def load_geographic_data(filename: str) -> tuple[list[ΦΛLine], bool]:
	""" load a bunch of polylines from a shapefile
	    :param filename: valid values include "admin_0_countries", "coastline", "lakes",
	                     "land", "ocean", "rivers_lake_centerlines"
	    :return: the coordinates along the border of each part (degrees), and a bool indicating
	             whether this is an open polyline as opposed to a closed polygon
	"""
	lines: list[ΦΛLine] = []
	closed = True
	with shapefile.Reader(f"../data/ne_110m_{filename}.zip") as shape_f:
		print(f"this {filename} file has {len(shape_f)} shapes")
		for shape in shape_f.shapes():
			closed = shape.shapeTypeName == "POLYGON"
			for i in range(len(shape.parts)):
				start = shape.parts[i]
				end = shape.parts[i + 1] if i + 1 < len(shape.parts) else len(shape.points)
				lines.append(np.empty(end - start, dtype=ΦΛPoint))
				for j, (λ, ф) in enumerate(shape.points[start:end]):
					lines[-1][j] = (max(-90, min(90, ф)), max(-180, min(180, λ)))
	return lines, closed


class Section:
	def __init__(self, ф_nodes: NDArray[float], λ_nodes: NDArray[float],
	             xy_nodes: NDArray[XYPoint], border: ΦΛLine):
		""" one lobe of an Elastik projection, containing a grid of latitudes and
		    longitudes as well as the corresponding x and y coordinates
		    :param ф_nodes: the node latitudes (deg)
		    :param λ_nodes: the node longitudes (deg)
		    :param xy_nodes: the grid of x- and y-values at each ф and λ (km)
		    :param border: the path that encloses the region this section defines (d"eg)
		"""
		self.x_projector = interpolate.RegularGridInterpolator(
			(ф_nodes, λ_nodes), xy_nodes["x"])
		self.y_projector = interpolate.RegularGridInterpolator(
			(ф_nodes, λ_nodes), xy_nodes["y"])
		self.border = border


	def get_planar_coordinates(self, points: NDArray[ΦΛPoint]
	                           ) -> NDArray[XYPoint]:
		""" take a point on the sphere and smoothly interpolate it to x and y """
		result = np.empty(points.shape, dtype=XYPoint)
		result["x"] = self.x_projector((points["latitude"], points["longitude"]))
		result["y"] = self.y_projector((points["latitude"], points["longitude"]))
		return result


	def contains(self, points: NDArray[ΦΛPoint]) -> NDArray[bool]:
		nearest_segment = np.full(points.shape, np.inf)
		ф, λ = points["latitude"], points["longitude"]
		inside = np.full(np.shape(ф), False)
		for i in range(1, self.border.shape[0]):
			ф0, λ0 = self.border[i - 1]
			ф1, λ1 = self.border[i]
			if λ1 == λ0:
				continue
			elif abs(λ1 - λ0) <= 180:
				straddles = (λ0 <= λ) != (λ1 <= λ)
			else:
				assert abs(λ0) == 180 and λ1 == -λ0 and ф0 == ф1
				straddles = abs(λ) == 180
				λ0, λ1 = λ1, λ0
			фX = (λ - λ0)/(λ1 - λ0)*(ф1 - ф0) + ф0
			distance = np.where(straddles, abs(фX - ф), np.inf)
			inside = np.where(distance < nearest_segment, (λ1 > λ0) != (фX > ф), inside)
			nearest_segment = np.minimum(nearest_segment, distance)
		return inside


if __name__ == "__main__":
	create_map(name="seal-ranges",
	           projection="mar",
	           duplicate=False,
	           background_style=dict(
		           facecolor="#fff",
	           ),
	           border_style=dict(
		           edgecolor="#000",
		           linewidth=1,
	           ),
	           data=[
		           ("ocean", dict(
			           facecolor="#77f",
			           edgecolor="none",
		           )),
		           ("coastline", dict(
			           color="#007",
			           linewidth=1,
		           )),
		           ("rivers_lake_centerlines", dict(
			           color="#007",
			           linewidth=.5,
		           ))
	           ])
	plt.show()
