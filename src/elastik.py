#!/usr/bin/env python
"""
elastik.py

this script is an example of how to use the Elastic projections. enclosed in this file is
everything you need to load the mesh files and project data onto them.
"""
from __future__ import annotations

from typing import Any

import h5py
import numpy as np
import shapefile
from matplotlib import pyplot as plt
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
			for i, line in enumerate(projected_data):
				plt.fill(line["x"], line["y"], **style)  # TODO: use a PathPatch instead of .fill() so there can be holes
		else:
			for i, line in enumerate(projected_data):
				plt.plot(line["x"], line["y"], **style)

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
		# for each line, project it into each section that can accommodate it
		for section in projection:

			# see which parts will project into the section and which are out of it
			points = np.array(line, dtype=ΦΛPoint)
			in_this_section = section.contains(points)
			j = np.arange(in_this_section.size)
			if np.any(in_this_section):
				# look for the points where it enters or exits the section
				exeunts = np.nonzero(in_this_section[j - 1] & ~in_this_section[j])[0]
				if exeunts.size > 0:
					# roll this so we can always assume exeunts[k] < entrances[k]
					points = np.roll(points, -exeunts[0])
					in_this_section = np.roll(in_this_section, -exeunts[0])
					exeunts -= exeunts[0]
					entrances = np.nonzero(~in_this_section[j - 1] & in_this_section[j])[0]
					# then set some precisely interpolated points at the interfaces
					trimd_points = np.empty(0, dtype=ΦΛPoint)
					for k in range(entrances.size):
						exeunt_index, entrance_index = exeunts[k], entrances[k]
						exeunt_point = find_intersection(
							points[[exeunt_index - 1, exeunt_index]],
							section.border)
						entrance_point = find_intersection(
							points[[entrance_index - 1, entrance_index]],
							section.border)
						print(entrance_point["latitude"], entrance_point["longitude"])
						if abs(entrance_point["longitude"]) > 180:
							raise
						next_exeunt_index = exeunts[k + 1] if k + 1 < exeunts.size else None
						trimd_points = np.concatenate([
							trimd_points,
							[exeunt_point, entrance_point],
							points[entrance_index:next_exeunt_index]])

					points = trimd_points

				# finally, project to the plane and save the result
				projected.append(section.get_planar_coordinates(points))
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


def find_intersection(short_path: ΦΛLine, long_path: ΦΛLine) -> ΦΛPoint:
	""" find the first point along long_path that intersects with short_path """
	for i in range(1, long_path.shape[0]):
		a = long_path[i - 1]
		b = long_path[i]
		for j in range(1, short_path.shape[0]):
			c = short_path[j - 1]
			d = short_path[j]
			Φ, Λ = "latitude", "longitude"
			denominator = ((a[Φ] - b[Φ])*(c[Λ] - d[Λ]) - (a[Λ] - b[Λ])*(c[Φ] - d[Φ]))
			if denominator == 0:
				continue
			intersection = np.empty((), dtype=ΦΛPoint)
			for k in [Φ, Λ]:
				intersection[k] = ((a[Φ]*b[Λ] - a[Λ]*b[Φ])*(c[k] - d[k]) -
				                   (a[k] - b[k])*(c[Φ]*d[Λ] - c[Λ]*d[Φ]))/\
				                  denominator
			k = Φ if c[Φ] != d[Φ] else Λ
			if min(c[k], d[k]) <= intersection[k] <= max(c[k], d[k]):
				intersection["latitude"] = max(-90, min(90, intersection["latitude"]))  # these two lines are because it’s slightly numericly unstable
				intersection["longitude"] = max(-180, min(180, intersection["longitude"]))
				return intersection
	raise ValueError(f"no intersection was found.")


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
		print((points["longitude"].min(), points["longitude"].max()))
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
