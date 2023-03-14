#!/usr/bin/env python
"""
run_all_scripts.py

generate all three elastik map projecitons after regenerating all of their prerequisites
"""
import time

from matplotlib import pyplot as plt

from build_mesh import build_mesh
from calculate_weights import calculate_weights
from create_map_projection import create_map_projection


if __name__ == "__main__":
	print(time.strftime("%Y-%m-%d %H:%M:%S"))
	calculate_weights(10, 0.5)
	for mesh in ["basic", "oceans", "mountains"]:
		build_mesh(mesh, 25)
	for projection in ["continents", "oceans", "countries"]:
		create_map_projection(projection)
	print(time.strftime("%Y-%m-%d %H:%M:%S"))
	plt.show()
