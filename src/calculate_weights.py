#!/usr/bin/env python
"""
calculate_weights.py

generate simple maps of importance as a function of location, to use when optimizing map
projections
"""
from math import nan, isnan, hypot, radians

import numpy as np
import shapefile
import tifffile
from matplotlib import pyplot as plt
from numpy.typing import NDArray

from util import bin_centers, to_cartesian, inside_region

FUDGE_FACTOR = 4 # some extra padding to put on the contiguus joints of the sections
# latitude of southernmost settlement
ANTARCTIC_CUTOFF = -56.
# latitude of northernmost settlement
ARCTIC_CUTOFF = 78.
# coordinates of sahara, australian, and canadian desert ellipses
DESERT_ELLIPSES = [(23, 8, 7, 20), (-24, 132, 7, 15), (90, -90, 35, 50)]
# coordinates of small, remote, uninhabited islands
EXCLUDED_ISLANDS = [(-6, 72), # chagos islands
                    (-54, 3), # bouvet island
                    (-40, -9), # gough island
                    (-46, 38), # prince edward island
                    (-53, 73), # heard island
                    # (-49, 69), # kerguelen islands
                    (-46, 52), # crozet islands
                    (15, 169), # bokak atoll
                    (10, -109), # clipperton island
                    (24, 154), # marcus island
                    (19, 167), # wake island
                    (1, -29), # sao pedro and sao paul
                    (-21, -29), # trindade and martim vaz
                    (-55, 159), # macquarie island
                    (-38, 78), # st. paul and amsterdam islands
                    (-49, 179)] # antipode and bounty islands


def load_coast_vertices(precision: float) -> list[tuple[float, float]]:
	""" load the coastline shapefiles, including as many islands as we can, but not being
	    too precise about the coastlines' exact shapes.
	"""
	excluded_ф, excluded_λ = np.transpose(EXCLUDED_ISLANDS)
	points = []
	λ_last, ф_last = nan, nan
	for data in ["coastline", "minor_islands_coastline"]:
		with shapefile.Reader(f"../data/ne_10m_{data}.zip") as shape_f:
			for shape in shape_f.shapes():
				for λ, ф in shape.points:
					edge_length = hypot(
						ф - ф_last, (λ - λ_last)*np.cos(radians((ф + ф_last)/2)))
					if isnan(λ_last) or edge_length > precision:
						λ_last, ф_last = λ, ф
						if not np.any((abs(ф - excluded_ф) < 2) & (abs(λ - excluded_λ) < 2)):
							points.append((ф, λ)) # exclude the chagos, prince edward, and bouvet islands because they're awkwardly situated

	return points


def uninhabited(ф: NDArray[float], λ: NDArray[float], desert_counts_as_uninhabited: bool) -> NDArray[bool]:
	""" return a bool array indicating which points are uninhabited by humans """
	uninhabited = np.full(np.broadcast(ф, λ).shape, False)
	for ф_0, λ_0 in EXCLUDED_ISLANDS:
		uninhabited |= ((abs(ф - ф_0) < 2) & (abs(λ - λ_0) < 2))
	if desert_counts_as_uninhabited:
		for ф_0, λ_0, a, b in DESERT_ELLIPSES:
			uninhabited |= (((ф - ф_0)/a)**2 + ((λ - λ_0)/b)**2 < 1)
		uninhabited |= (ф >= ARCTIC_CUTOFF) | (ф <= ANTARCTIC_CUTOFF)
	return uninhabited


def calculate_coast_distance(ф: NDArray[float], λ: NDArray[float], coast: list[tuple[float, float]],
                             section: NDArray[float], exclude_antarctica: bool) -> NDArray[float]:
	""" take a set of latitudes and longitudes and calculate the angular distance from
	    each point to the nearest shoreline (in degrees)
	"""
	фф, λλ = np.meshgrid(ф, λ, indexing="ij")
	xx, yy, zz = to_cartesian(фф, λλ)

	# first crop the coasts inside this section
	points = np.array(coast)
	points = points[inside_region(points[:, 0], points[:, 1], section), :]
	points = points[~uninhabited(points[:, 0], points[:, 1], exclude_antarctica), :]
	minimum_distance = np.full((ф.size, λ.size), np.inf)

	# then calculate the distances
	for ф_0, λ_0 in points:
		x_0, y_0, z_0 = to_cartesian(ф_0, λ_0)
		minimum_distance = np.minimum(minimum_distance,
		                              np.degrees(np.arccos(x_0*xx + y_0*yy + z_0*zz)))
	return minimum_distance


def load_cut_file(filename: str) -> list[NDArray[float]]:
	""" load the borders of the sections for a given map projection """
	cut_data = np.loadtxt(filename)
	section_indices = np.nonzero(np.all(cut_data == cut_data[0, :], axis=1))[0]
	section_indices = np.concatenate([section_indices, [None]])
	cuts = []
	for h in range(len(section_indices) - 1):
		cuts.append(cut_data[section_indices[h]:section_indices[h + 1]])
	margin = FUDGE_FACTOR if cut_data[0][0] > 0 else -FUDGE_FACTOR
	sections = []
	for h in range(len(section_indices) - 1):
		sections.append(np.concatenate([
			[cuts[h - 1][-1, :] + [0, margin]],
			cuts[h - 1][::-1],
			cuts[h],
			[cuts[h][-1, :] - [0, margin]],
		]))
	return sections


def load_land_polygons() -> list[list[tuple[float, float]]]:
	""" load the land polygon shapefile (110m resolution) """
	polygons = []
	with shapefile.Reader(f"../data/ne_110m_land.zip") as shape_f:
		for shape in shape_f.shapes():
			polygons.append(shape.points)
	return polygons


def find_land_mask(ф_grid: NDArray[float], λ_grid: NDArray[float], exclude_antarctica: bool) -> NDArray[bool]:
	""" bild a 2D bool array that is True for coordinates on land and False for coordinates in the ocean. """
	crossings = np.full((ф_grid.size, λ_grid.size), 0)
	polygons = load_land_polygons()
	for polygon in polygons:
		for i in range(len(polygon)):
			λ_0, ф_0 = polygon[i - 1]
			λ_1, ф_1 = polygon[i]
			if λ_0 == λ_1:
				continue
			ф_X = ф_0 + (λ_grid - λ_0)/(λ_1 - λ_0)*(ф_1 - ф_0)
			intersects = np.not_equal(λ_0 < λ_grid, λ_1 < λ_grid)
			ф_X[~intersects] = np.inf
			crossings[ф_X[np.newaxis, :] < ф_grid[:, np.newaxis]] += 1
	ф_grid, λ_grid = np.meshgrid(ф_grid, λ_grid, indexing="ij", sparse=True)
	return np.where(uninhabited(ф_grid, λ_grid, exclude_antarctica),
	                False, crossings%2 == 1)


def calculate_weights(coast_width: float, precision: float):
	# define the grid
	ф_edges = bin_centers(np.linspace(-90, 90, 188)) # (these are intentionally weerd numbers to reduce roundoff issues)
	λ_edges = bin_centers(np.linspace(-180, 180, 375))
	ф, λ = bin_centers(ф_edges), bin_centers(λ_edges)

	# load the coast data
	coast_vertices = load_coast_vertices(precision)

	# iterate thru the four weight files we want to generate
	for crop_antarctica in [False, True]:
		# load the land data with or without antarctica
		land = find_land_mask(ф, λ, crop_antarctica)

		for cut_file, value_land in [("basic", True), ("oceans", True), ("mountains", False)]:
			# load the cut file
			sections = load_cut_file(f"../spec/cuts_{cut_file}.txt")

			# get the set of points that are uniformly important
			global_mask = land if value_land else ~land

			for h, section in enumerate(sections):
				filename = f"../spec/pixels_{cut_file}_{h}{'_land' if value_land else '_sea'}{'_sinATA' if crop_antarctica else ''}.tif"
				print(filename)

				# get the distance of each point from the nearest contained coast
				coast_distance = calculate_coast_distance(ф, λ, coast_vertices, section, crop_antarctica)

				# get the points on the mesh inside this section
				in_section = inside_region(ф, λ, section)

				# combine everything
				mask = global_mask & in_section
				importance = np.where(mask, 1, np.maximum(0, 1 - coast_distance/coast_width)**2)

				# save and plot
				tifffile.imwrite(filename, importance)
				plt.figure(f"Weightmap {cut_file}-{h}-{2*crop_antarctica + value_land}")
				plt.pcolormesh(λ_edges, ф_edges, importance)
				plt.plot(section[:, 1], section[:, 0], f"k", linewidth=1)
				plt.plot(section[:, 1], section[:, 0], f"w--", linewidth=1)
				plt.axis([λ_edges[0], λ_edges[-1], ф_edges[0], ф_edges[-1]])


if __name__ == "__main__":
	calculate_weights(coast_width=10, precision=0.5)
	plt.show()
