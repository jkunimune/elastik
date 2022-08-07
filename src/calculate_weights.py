#!/usr/bin/env python
"""
calculate_weights.py

generate simple maps of importance as a function of location, to use when optimizing map
projections
"""
import math
import os

import numpy as np
import shapefile
import tifffile
from matplotlib import pyplot as plt

from util import bin_centers, to_cartesian


def load_coast_vertices(precision) -> list[tuple[float, float]]:
	""" load the coastline shapefiles, including as many islands as we can, but not being
	    too precise about the coastlines' exact shapes.
	"""
	points = []
	λ_last, ф_last = np.nan, np.nan
	for data in ["coastline", "minor_islands_coastline"]:
		with shapefile.Reader(f"../data/ne_10m_{data}.zip") as shape_f:
			for shape in shape_f.shapes():
				for λ, ф in shape.points:
					edge_length = math.hypot(
						ф - ф_last, (λ - λ_last)*np.cos(math.radians((ф + ф_last)/2)))
					if math.isnan(λ_last) or edge_length > precision:
						if abs(ф + 6) > 2 or abs(λ - 72) > 2: # exclude the chagos archipelago because it's awkwardly situated
							points.append((λ, ф))
						λ_last, ф_last = λ, ф
	return points


def calculate_coast_distance(ф: np.ndarray, λ: np.ndarray, precision: float, southern_cutoff: float) -> np.ndarray:
	""" take a set of latitudes and longitudes and calculate the angular distance from
	    each point to the nearest shoreline (in degrees)
	"""
	фф, λλ = np.meshgrid(ф, λ, indexing="ij")
	xx, yy, zz = to_cartesian(фф, λλ)

	points = load_coast_vertices(precision)
	minimum_distance = np.full((ф.size, λ.size), np.inf)

	for λ_0, ф_0 in points: # TODO: ideally it would generate the weits for a given mesh
		if southern_cutoff is None or ф_0 > southern_cutoff:
			x_0, y_0, z_0 = to_cartesian(ф_0, λ_0)
			minimum_distance = np.minimum(minimum_distance,
			                              np.degrees(np.arccos(x_0*xx + y_0*yy + z_0*zz)))
	return minimum_distance


def load_land_polygons() -> list[list[tuple[float, float]]]:
	""" load the land polygon shapefile (110m resolution) """
	polygons = []
	with shapefile.Reader(f"../data/ne_110m_land.zip") as shape_f:
		for shape in shape_f.shapes():
			polygons.append(shape.points)
	return polygons


def find_land_mask(ф_grid: np.ndarray, λ_grid: np.ndarray, southern_cutoff: float) -> np.ndarray:
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
	if southern_cutoff is not None:
		crossings[ф_grid <= southern_cutoff] = 0
	return crossings%2 == 1


def calculate_weights(coast_width: float, precision: float, antarctic_cutoff=-56.):
	# define the 1° grid
	ф_edges = bin_centers(np.linspace(-90, 90, 181))
	λ_edges = bin_centers(np.linspace(-180, 180, 361))
	ф, λ = bin_centers(ф_edges), bin_centers(λ_edges)

	# load some cuts, just for reference
	cut_sets = []
	for filename in os.listdir("../spec"):
		if filename.startswith("cuts_"):
			cut_data = np.loadtxt(os.path.join("../spec", filename))
			section_indices = np.nonzero(np.all(cut_data == cut_data[0, :], axis=1))[0]
			section_indices = np.concatenate([section_indices, [None]])
			cut_set = []
			for i in range(len(section_indices) - 1):
				cut_set.append(cut_data[section_indices[i]:section_indices[i + 1]])
			cut_sets.append(cut_set)

	# iterate thru the four weight files we want to generate
	for crop_antarctica in [False, True]:
		# load the land data with or without antarctica
		cutoff = antarctic_cutoff if crop_antarctica else None
		land = find_land_mask(ф, λ, cutoff)

		# get the distance of each point from the nearest coast
		coast_distance = calculate_coast_distance(ф, λ, precision, cutoff)

		for value_land in [False, True]:
			filename = f"../spec/pixels{'_land' if value_land else '_sea'}{'_sinATA' if crop_antarctica else ''}.tif"
			print(filename)

			# get the set of points that are uniformly important
			mask = land if value_land else ~land

			# combine them
			importance = np.where(mask, 1, np.maximum(0, 1 - (coast_distance/coast_width)**2))

			# save and plot
			tifffile.imwrite(filename, importance)
			plt.figure(f"Weightmap {2*crop_antarctica + value_land}")
			plt.pcolormesh(λ_edges, ф_edges, importance)
			for cut_set in cut_sets: # show the cut systems, just for reference
				for cut in cut_set:
					plt.plot(cut[:, 1], cut[:, 0], f"k", linewidth=1)
				plt.scatter([cut[-1, 1] for cut in cut_set], [cut[-1, 0] for cut in cut_set], c=f"k", s=10)
			plt.axis([λ_edges[0], λ_edges[-1], ф_edges[0], ф_edges[-1]])


if __name__ == "__main__":
	calculate_weights(coast_width=4.5, precision=.5)
	plt.show()
