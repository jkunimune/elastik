import os

import numpy as np
from matplotlib.colors import ListedColormap

CUSTOM_CMAP: dict[str, ListedColormap] = {}

for filename in os.listdir("../resources/colormaps"):
	cmap_data = np.loadtxt(os.path.join("../resources/colormaps", filename), delimiter=",")
	cmap_name = filename[:-4]
	CUSTOM_CMAP[cmap_name] = ListedColormap(cmap_data, name=cmap_name)
