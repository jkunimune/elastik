#!/usr/bin/env python
"""
create_example_maps.py

this script is an example of how to use the Elastic projections. enclosed in this file is
everything you need to load the mesh files and project data onto them.
"""
from __future__ import annotations

from math import nan, sqrt
from typing import Any, Optional

import h5py
import numpy as np
import shapefile
from matplotlib import pyplot as plt, path
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from numpy.typing import NDArray
from scipy import interpolate


# TYPE DEFINITIONS

Style = dict[str, Any]
XYPoint = np.dtype([("x", float), ("y", float)])
XYLine = NDArray[XYPoint]
XYFeature = tuple[int, float, list[XYLine]]
ΦΛPoint = np.dtype([("latitude", float), ("longitude", float)])
ΦΛLine = NDArray[ΦΛPoint]
ΦΛFeature = tuple[int, float, list[ΦΛLine]]


# GLOBAL CONSTANTS

SNIPPING_LENGTH = 1000


# FUNCTIONS RELATING TO GRAPHICS AND DATA

def create_example_maps():
	create_map(name="political",
	           projection="elastic-III",
	           background_style=dict(
		           facecolor="#ebf8ff",
	           ),
	           border_style=dict(
		           edgecolor="#000",
		           linewidth=0.6,
	           ),
	           data=[
		           ("ne_110m_admin_0_countries", dict(
			           edgecolor="#000",
			           linewidth=0.3,
			           facecolor=[
				           "#c799b5", "#d6a4b7", "#e3afb9", "#cf9f9d", "#daac9e", "#e2b9a0",
				           "#caab88", "#ceba8e", "#b3ab7c", "#b5ba87", "#b7c994", "#9bb987",
				           "#9bc797", "#9cd5a8", "#81c49e", "#83d1b1", "#68bfa7", "#6dcbbb",
				           "#75d7cf", "#60c4c5", "#6dced8", "#5ebacd", "#70c4df", "#83cdf0",
				           "#79b8e2", "#8ec0f1", "#a3c9ff", "#9ab4ed", "#aebcf8", "#a4a7e2",
				           "#b7afeb", "#cab8f2", "#bda4da", "#ceaddf", "#dfb7e3", "#d0a4c9",
			           ],
		           )),
	           ])
	create_map(name="biomes",
	           projection="elastic-I",
	           background_style=dict(
		           facecolor="#0f1d8e",
	           ),
	           border_style=dict(
		           edgecolor="none",
	           ),
	           data=[
		           ("wwf_teow", dict(
			           edgecolor="facecolor",
			           linewidth=0.1,
			           facecolor=[
				           "#2d540e",  # 1: tropical moist broadleaf forest
				           "#425c18",  # 2: tropical dry broadleaf forest
				           "#29622b",  # 3: tropical conifer forest
				           "#487e35",  # 4: temperate broadleaf forest
				           "#29622b",  # 5: temperate conifer forest
				           "#1d5b30",  # 6: polar conifer forest (taiga)
				           "#6b7b30",  # 7: tropical grassland
				           "#87904f",  # 8: temperate grassland
				           "#7c9045",  # 9: flooded grassland
				           "#bda575",  # 10: mountain grassland
				           "#ffffff",  # 11: polar grassland (tundra)
				           "#9ba96b",  # 12: mediterranean forest
				           "#fbeaae",  # 13: desert
				           "#336a22",  # 14: mangrove
				           "#ffffff",  # 99: rock and ice
			           ],
		           )),
		           ("ne_10m_lakes", dict(
			           edgecolor="none",
			           facecolor="#0f1d8e",
		           ))
	           ])
	create_map(name="water",
	           projection="elastic-II",
	           background_style=dict(
		           facecolor="#fff",
	           ),
	           border_style=dict(
		           edgecolor="none",
	           ),
	           data=[
		           ("ne_50m_ocean", dict(
			           facecolor="#8bf3f9",
			           edgecolor="none",
		           )),
		           ("ne_10m_bathymetry_K_200", dict(
			           facecolor="#63dcf2",
			           edgecolor="none",
		           )),
		           ("ne_10m_bathymetry_J_1000", dict(
			           facecolor="#44c3ec",
			           edgecolor="none",
		           )),
		           ("ne_10m_bathymetry_I_2000", dict(
			           facecolor="#28aae7",
			           edgecolor="none",
		           )),
		           ("ne_10m_bathymetry_H_3000", dict(
			           facecolor="#0c92e4",
			           edgecolor="none",
		           )),
		           ("ne_10m_bathymetry_G_4000", dict(
			           facecolor="#1878e0",
			           edgecolor="none",
		           )),
		           ("ne_10m_bathymetry_F_5000", dict(
			           facecolor="#335dcd",
			           edgecolor="none",
		           )),
		           ("ne_10m_bathymetry_E_6000", dict(
			           facecolor="#3c47a9",
			           edgecolor="none",
		           )),
		           ("ne_10m_bathymetry_D_7000", dict(
			           facecolor="#363680",
			           edgecolor="none",
		           )),
		           ("ne_10m_bathymetry_C_8000", dict(
			           facecolor="#292657",
			           edgecolor="none",
		           )),
		           ("ne_10m_bathymetry_B_9000", dict(
			           facecolor="#191534",
			           edgecolor="none",
		           )),
		           ("ne_10m_bathymetry_A_10000", dict(
			           facecolor="#060413",
			           edgecolor="none",
		           )),
		           ("ne_50m_coastline", dict(
			           color="#009",
			           linewidth=0.2,
		           )),
		           ("ne_50m_rivers_lake_centerlines_scale_rank", dict(
			           color="#009",
			           linewidth=0,
		           )),
		           ("ne_110m_lakes", dict(
			           facecolor="#8bf3f9",
			           edgecolor="#009",
			           linewidth=0.2,
		           )),
	           ])


def create_map(name: str, projection: str,
               background_style: Style, border_style: Style,
               data: list[tuple[str, Style]]):
	""" draw a world map in a MatPlotLib figure using an elastic projection and some datasets.
	    :param name: the name with which to save the figure
	    :param projection: the name of the elastic projection to use
	    :param background_style: the Style for the otherwise unfilled regions of the map area
	    :param border_style: the Style for the line drawn around the complete map area
	    :param data: a list of elements to include in the map
	"""
	print(f"creating the {name} map.")
	sections, boundary, aspect_ratio = load_elastic_projection(projection)

	fig = plt.figure(name, figsize=(6*sqrt(aspect_ratio), 6/sqrt(aspect_ratio)))
	ax = fig.subplots()

	ax.fill(boundary["x"], boundary["y"], edgecolor="none", **background_style)

	for i, (data_name, style) in enumerate(data):
		print(f"adding {data_name} to the map...")
		zorder = 1 + i
		multiple_colors = "facecolor" in style and type(style["facecolor"]) is list
		multiple_widths = "linewidth" in style and style["linewidth"] == 0
		unprojected_data, closed = load_geographic_data(data_name)
		projected_data = project(unprojected_data, sections)
		projected_data = cut_lines_that_cross_interruptions(projected_data, closed)
		if closed:
			for category, width, lines in projected_data:
				feature_specific_style = {**style}
				if multiple_colors:
					color_index = category%len(style["facecolor"])
					feature_specific_style["facecolor"] = style["facecolor"][color_index]
				if style["edgecolor"] == "facecolor":
					feature_specific_style["edgecolor"] = feature_specific_style["facecolor"]
				points: list[tuple[float, float]] = []
				codes: list[int] = []
				for line in lines:
					for k, point in enumerate(line):
						points.append((point["x"], point["y"]))
						codes.append(Path.MOVETO if k == 0 else Path.LINETO)
					points.append((nan, nan))
					codes.append(Path.CLOSEPOLY)
				path = Path(points, codes)  # type: ignore
				patch = PathPatch(path, zorder=zorder, **feature_specific_style)
				ax.add_patch(patch)
		else:
			for category, width, lines in projected_data:
				feature_specific_style = {**style}
				if multiple_widths:
					feature_specific_style["linewidth"] = width
				for line in lines:
					ax.plot(line["x"], line["y"], zorder=zorder, **feature_specific_style)

	ax.fill(boundary["x"], boundary["y"], facecolor="none", **border_style)

	ax.axis("equal")
	ax.margins(.01)
	ax.axis("off")
	fig.savefig(f"../examples/{name}.png", dpi=300,
	            bbox_inches="tight", pad_inches=0)
	print(f"saved the {name} map!")


def cut_lines_that_cross_interruptions(features: list[XYFeature], closed: bool) -> list[XYFeature]:
	""" if you naively project lines on a map projection that are not pre-cut at the interruptions,
	    you’ll get a lot of extraneous lines crisscrossing the map.  this function deals with that
	    problem by cutting any lines that seem suspiciusly long.
	    :param features: the data to investigate and adjust
	    :param closed: whether to worry about forming closed paths from the continuus regions of each line
	"""
	new_features: list[XYFeature] = []
	for category, width, lines in features:
		new_lines: list[XYLine] = []
		new_lines_to_be_merged: list[XYLine] = []
		for line in lines:
			# start by finding line segments longer than 100 km
			j = np.arange(0 if closed else 1, line.size)
			lengths = np.hypot(line["x"][j] - line["x"][j - 1],
			                   line["y"][j] - line["y"][j - 1])
			cuts = j[np.nonzero(lengths > SNIPPING_LENGTH)[0]]
			# don’t try to do anything if there are not cuts
			if cuts.size == 0:
				new_lines.append(line)
			else:
				if closed:
					# cycle the points based on the discontinuities you found so it starts and ends with one
					line = np.roll(line, -cuts[0])
					cuts = np.concatenate([cuts - cuts[0], [line.size]])
				else:
					cuts = np.concatenate([[0], cuts, [line.size]])
				sections: list[XYLine] = []
				# break the input up into separate continuus segments
				for k in range(1, cuts.size):
					sections.append(line[cuts[k - 1]:cuts[k]])
				# remove any length 1 sections
				for k in range(len(sections) - 1, -1, -1):
					if len(sections[k]) <= 1:
						sections.pop(k)
				# don’t mark them as finalized yet if we need to close the paths
				if closed:
					new_lines_to_be_merged += sections
				else:
					new_lines += sections

		# if it’s closed, you have to reorder the lines so they go together
		new_lines_to_be_merged = sorted(new_lines_to_be_merged, key=lambda feature: len(feature[1]))
		# go thru all of these lines that are not finalized
		if len(new_lines_to_be_merged) > 0:
			merged_line: Optional[XYLine] = None
			while True:
				if merged_line is not None:
					# take the endpoint of you current line
					endpoint = merged_line[-1]
					potential_next_startpoints = np.array(
						[line[0] for line in new_lines_to_be_merged] + [merged_line[0]])
					# and find the pending startpoint closest to it
					next_index = np.argmin(np.hypot(potential_next_startpoints["x"] - endpoint["x"],
					                                potential_next_startpoints["y"] - endpoint["y"]))
					# if that nearest startpoint is its own, close it and finish it
					if next_index == len(new_lines_to_be_merged):
						new_lines.append(merged_line)
						merged_line = None
					# if the nearest startpoint is a different segment, merge them
					else:
						next_line = new_lines_to_be_merged.pop(next_index)
						merged_line = np.concatenate([merged_line, next_line])
				else:
					if len(new_lines_to_be_merged) > 0:
						# arbitrarily take the next pending line whenever we need to restart
						merged_line = new_lines_to_be_merged.pop()
					else:
						# stop when we run out of lines to be merged
						break
		new_features.append((category, width, new_lines))
	return new_features


def load_geographic_data(filename: str) -> tuple[list[ΦΛFeature], bool]:
	""" load a bunch of polylines from a shapefile
	    :param filename: the name of the shapefile zip file
	    :return: a list of features, each comprising a "category" (the biome if available, the index
	             otherwise), a "width" (only available from Natural Earth’s "rivers with scale
	             ranks" dataset), and a list of series of geographic coordinates (degrees);
	             and a bool indicating whether this is a closed polygon rather than an open polyline
	"""
	encoding = "latin-1" if "wwf_" in filename else "utf-8"
	features: list[ΦΛFeature] = []
	closed = True
	with shapefile.Reader(f"../resources/shapefiles/{filename}.zip", encoding=encoding) as f:
		for index, (record, shape) in enumerate(zip(f.records(), f.shapes())):
			closed = shape.shapeTypeName == "POLYGON"
			try:
				category = min(15, int(record["BIOME"])) - 1
			except IndexError:
				category = index
			try:
				width = 0.5*record["strokeweig"]
			except IndexError:
				width = 1
			lines: list[XYLine] = []
			for i in range(len(shape.parts)):
				start = shape.parts[i]
				end = shape.parts[i + 1] if i + 1 < len(shape.parts) else len(shape.points)
				line = np.empty(end - start, dtype=ΦΛPoint)
				for j, (λ, ф) in enumerate(shape.points[start:end]):
					line[j] = (max(-90, min(90, ф)), max(-180, min(180, λ)))
				lines.append(line)
			features.append((category, width, lines))
	return features, closed


# FUNCTIONS RELATED TO THE ELASTIC PROJECTIONS

def project(features: list[ΦΛFeature], projection: list[Section]) -> list[XYFeature]:
	""" apply the given Elastic projection, defined by a list of sections, to the given series of
	    latitudes and longitudes.
	"""
	projected_features: list[XYFeature] = []
	for j, (category, width, lines) in enumerate(features):
		print(f"projecting feature {j: 3d}/{len(features): 3d} ({sum(len(line) for line in lines)} points)")
		projected_lines: list[XYLine] = []
		for line in lines:
			projected_line = np.empty(line.size, dtype=XYPoint)
			projected_line[:] = (nan, nan)
			# for each line, project it into whichever section that can accommodate it
			for section in projection:
				in_this_section = section.contains(line)
				projected_line[in_this_section] = section.get_planar_coordinates(line[in_this_section])
			# check that each point was projected by at least one section
			assert not np.any(np.isnan(projected_line["x"]))
			projected_lines.append(projected_line)
		projected_features.append((category, width, projected_lines))
	print(f"projected {len(projected_features)} features!")
	return projected_features


def load_elastic_projection(name: str) -> tuple[list[Section], XYLine, float]:
	""" load the hdf5 file that defines an elastic projection
	    :param name: one of "elastic-I", "elastic-II", or "elastic-III"
	    :return: the list of sections that comprise this projection, and the map’s full projected outer shape
	"""
	with h5py.File(f"../projection/{name}.h5", "r") as file:
		sections = []
		for h in range(file.attrs["number of sections"]):
			sections.append(Section(file[f"section {h}/latitude"][:],
			                        file[f"section {h}/longitude"][:],
			                        file[f"section {h}/projected points"][:, :],
			                        file[f"section {h}/boundary"][:],
			                        ))
		boundary = file["projected boundary"][:]
		aspect_ratio = (file["bounding box"]["x"][1] - file["bounding box"]["x"][0])/ \
		               (file["bounding box"]["y"][1] - file["bounding box"]["y"][0])
	return sections, boundary, aspect_ratio


class Section:
	def __init__(self, ф_nodes: NDArray[float], λ_nodes: NDArray[float],
	             xy_nodes: NDArray[XYPoint], border: ΦΛLine):
		""" one lobe of an Elastic projection, containing a grid of latitudes and
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
		self.border = path.Path(
			np.stack([border["latitude"], border["longitude"]], axis=-1)) # type: ignore
		self.border_is_counterclockwise = is_counterclockwise(self.border)


	def get_planar_coordinates(self, points: NDArray[ΦΛPoint]
	                           ) -> NDArray[XYPoint]:
		""" take a point on the sphere and smoothly interpolate it to x and y """
		result = np.empty(points.size, dtype=XYPoint)
		result["x"] = self.x_projector((points["latitude"], points["longitude"]))
		result["y"] = self.y_projector((points["latitude"], points["longitude"]))
		return result


	def contains(self, points: NDArray[ΦΛPoint]) -> NDArray[bool]:
		""" whether the given point is within this Section’s boundary """
		points = np.stack([points["latitude"], points["longitude"]], axis=-1)
		# make sure you check the border orientation, because Matplotlib won't
		if self.border_is_counterclockwise:  # use the radius parameter to ensure points on the boundary are counted
			return self.border.contains_points(points, radius=-1e-9) # type: ignore
		else:
			return ~self.border.contains_points(points, radius=1e-9) # type: ignore


def is_counterclockwise(path: path.Path) -> bool:
	""" determines whether the polygon is oriented in the normal direction """
	area = 0
	for i in range(len(path)):
		area += path.vertices[i - 1, 1]*path.vertices[i, 0] - \
		        path.vertices[i - 1, 0]*path.vertices[i, 1]
	return area > 0


if __name__ == "__main__":
	create_example_maps()
	plt.show()
