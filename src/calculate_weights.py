#!/usr/bin/env python
"""
calculate_weights.py

generate simple maps of importance as a function of location, to use when optimizing map
projections
"""
import math

import numpy as np
import shapefile
from matplotlib import pyplot as plt

from util import bin_centers, to_cartesian


COAST_WIDTH = 8 # degrees
PRECISION = 1 # degrees


def load_coast_vertices(precision) -> list[tuple[float, float]]:
	""" load the coastline shapefiles, including as many islands as we can, but not being
	    too precise about the ceastlines' exact shapes.
	"""
	points = []
	for data in ["coastline"]:#, "minor_islands_coastline"]:
		λ_last, ф_last = np.nan, np.nan
		with shapefile.Reader(f"../data/ne_10m_{data}.zip") as shapef:
			for shape in shapef.shapes():
				for λ, ф in shape.points:
					edge_length = math.hypot(
						ф - ф_last, (λ - λ_last)*np.cos(math.radians((ф + ф_last)/2)))
					if math.isnan(λ_last) or edge_length > precision:
						points.append((λ, ф))
						λ_last, ф_last = λ, ф
	return points


if __name__ == "__main__":
	ф_edges = bin_centers(np.linspace(-90, 90, 181))
	λ_edges = bin_centers(np.linspace(-180, 180, 361))
	ф, λ = bin_centers(ф_edges), bin_centers(λ_edges)
	фф, λλ = np.meshgrid(ф, λ, indexing="ij")
	xx, yy, zz = to_cartesian(фф, λλ)

	print("loading...")
	points = load_coast_vertices(PRECISION)
	minimum_distance = np.full((ф.size, λ.size), np.inf)

	print(len(points))
	for λ0, ф0 in points:
		x0, y0, z0 = to_cartesian(ф0, λ0)
		minimum_distance = np.minimum(minimum_distance, np.degrees(np.arccos(x0*xx + y0*yy + z0*zz)))

	importance = np.exp(-minimum_distance/COAST_WIDTH)

	plt.pcolormesh(λ_edges, ф_edges, importance)
	points = np.array(points)
	plt.scatter(points[:, 0], points[:, 1], c="w", s=1)
	plt.show()
