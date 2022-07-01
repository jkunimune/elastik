#!/usr/bin/env python
"""
calculate_weights.py

generate simple maps of importance as a function of location, to use when optimizing map
projections
"""
import math

import numpy as np
import shapefile
import tifffile
from matplotlib import pyplot as plt

from util import bin_centers, to_cartesian


COAST_WIDTH = 10 # degrees
PRECISION = .5 # degrees
ANTARCTIC_CUTOFF = -56 # degrees


def load_coast_vertices(precision) -> list[tuple[float, float]]:
	""" load the coastline shapefiles, including as many islands as we can, but not being
	    too precise about the ceastlines' exact shapes.
	"""
	points = []
	λ_last, ф_last = np.nan, np.nan
	for data in ["coastline", "minor_islands_coastline"]:
		with shapefile.Reader(f"../data/ne_10m_{data}.zip") as shapef:
			for shape in shapef.shapes():
				for λ, ф in shape.points:
					edge_length = math.hypot(
						ф - ф_last, (λ - λ_last)*np.cos(math.radians((ф + ф_last)/2)))
					if math.isnan(λ_last) or edge_length > precision:
						points.append((λ, ф))
						λ_last, ф_last = λ, ф
	return points


def calculate_coast_distance(ф: np.ndarray, λ: np.ndarray, crop_antarctica: bool) -> np.ndarray:
	""" take a set of latitudes and longitudes and calculate the angular distance from
	    each point to the nearest shoreline (in degrees)
	"""
	фф, λλ = np.meshgrid(ф, λ, indexing="ij")
	xx, yy, zz = to_cartesian(фф, λλ)

	points = load_coast_vertices(PRECISION)
	minimum_distance = np.full((ф.size, λ.size), np.inf)

	for λ_0, ф_0 in points:
		if not crop_antarctica or ф_0 > ANTARCTIC_CUTOFF:
			x_0, y_0, z_0 = to_cartesian(ф_0, λ_0)
			minimum_distance = np.minimum(minimum_distance,
			                              np.degrees(np.arccos(x_0*xx + y_0*yy + z_0*zz)))
	return minimum_distance


def load_land_polygons() -> list[list[tuple[float, float]]]:
	""" load the land polygon shapefile (110m resolution) """
	polygons = []
	with shapefile.Reader(f"../data/ne_110m_land.zip") as shapef:
		for shape in shapef.shapes():
			polygons.append(shape.points)
	return polygons


def find_land_mask(ф_grid: np.ndarray, λ_grid: np.ndarray, crop_antarctica: bool) -> np.ndarray:
	""" bild a 2D bool array that is True for coordinates on land and False for coordinates in the ocean. """
	crossings = np.full((ф.size, λ.size), 0)
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
	if crop_antarctica:
		crossings[ф_grid <= ANTARCTIC_CUTOFF] = 0
	return crossings%2 == 1


if __name__ == "__main__":
	ф_edges = bin_centers(np.linspace(-90, 90, 181))
	λ_edges = bin_centers(np.linspace(-180, 180, 361))
	ф, λ = bin_centers(ф_edges), bin_centers(λ_edges)

	# iterate thru the four weight files we want to generate
	for crop_antarctica in [False, True]:
		for value_land in [False, True]:
			filename = f"../spec/pixels{'_land' if value_land else '_sea'}{'_sen-ata' if crop_antarctica else ''}.tif"
			print(filename)

			# get the set of points that are uniformly important
			land = find_land_mask(ф, λ, crop_antarctica=crop_antarctica)
			mask = land if value_land else ~land

			# get the distance of each point from the nearest coast
			coast_distance = calculate_coast_distance(ф, λ, crop_antarctica=crop_antarctica)

			# combine them
			importance = np.where(mask, 1, np.exp(-coast_distance/COAST_WIDTH))

			# save and plot
			tifffile.imwrite(filename, (255.9*importance).astype(int))
			plt.figure()
			plt.pcolormesh(λ_edges, ф_edges, importance)

	plt.show()
