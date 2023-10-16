#!/usr/bin/env python
"""
run_all_scripts.py

generate all three elastic map projections after regenerating all of their prerequisites
"""
import time

from matplotlib import pyplot as plt

from build_mesh import build_mesh
from calculate_weights import calculate_weights
from create_map_projection import create_map_projection
from create_example_maps import create_example_maps

if __name__ == "__main__":
	print(time.strftime("%Y-%m-%d %H:%M:%S"))
	calculate_weights()
	plt.close("all")
	for mesh, resolution in [("basic", 24), ("oceans", 10), ("mountains", 18)]:
		build_mesh(mesh, resolution)
	plt.close("all")
	for projection in ["continents", "oceans", "countries"]:
		create_map_projection(projection)
	create_example_maps()

	print(time.strftime("%Y-%m-%d %H:%M:%S"))
	plt.show()
