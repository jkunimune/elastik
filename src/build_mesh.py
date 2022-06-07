"""
build_mesh.py

take an interruption file, and use it to generate and save a basic interrupted map
projection mesh that can be further optimized.
all angles are in radians. indexing is z[i,j] = z(ф[i], λ[j])
"""
import numpy as np
from matplotlib import pyplot as plt


# how many cells per degree
RESOLUTION = 0.1
# radius of earth in km
EARTH_RADIUS = 6370


class Node:
	def __init__(self, i: int, j: int, x: float, y: float):
		""" a point at which the relation between (ф, λ) and (x, y) is fixed """
		self.i: int = i
		self.j: int = j
		self.x: float = x
		self.y: float = y


	def __repr__(self) -> str:
		return f"Node({self.i}, {self.j}, {self.x:.0f}, {self.y:.0f})"


class Cell:
	def __init__(self, i_sw: int, j_sw: int, max_j: int,
	             sw: Node or None, se: Node or None,
	             ne: Node or None, nw: Node or None,
	             top: float, bottom: float, hite: float):
		""" a trapezoidal region formed by four adjacent Nodes. it keeps track of the
		    canonical lengths of each of its sides so that it can calculate its own
		    derivatives. the constructor will create and place new Nodes if some are None.
		    :param: i_sw the ф index of the southwestern Node
		    :param: j_sw the λ index of the southwestern Node
		    :param: sw the southwestern Node to use or None if we need to create one
		    :param: se the southeastern Node to use or None if we need to create one
		    :param: ne the northeastern Node to use or None if we need to create one
		    :param: nw the northwestern Node to use or None if we need to create one
		"""
		# calculate the positions of the Nodes
		nodes = [sw, se, ne, nw]
		if all(node is None for node in nodes):
			raise ValueError("at least two nodes must be specified")
		lengths = [bottom, hite, top, hite]
		# start by replacing any unspecified Nodes with lists
		for k in range(4):
			if nodes[k] is None:
				nodes[k] = []
		# then iterate over each edge
		for k in range(4):
			a = nodes[k]
			b = nodes[(k + 1)%4]
			# if the edge is fully defined
			if type(a) is Node and type(b) is Node:
				# use it to guess the best place for the opposite edge
				ab = np.hypot(a.x - b.x, a.y - b.y)
				if type(nodes[(k + 2)%4]) is list:
					bc = lengths[(k + 1)%4]
					c = (b.x + (a.y - b.y)/ab*bc, b.y - (a.x - b.x)/ab*bc)
					nodes[(k + 2)%4].append(c)
				if type(nodes[(k + 3)%4]) is list:
					da = lengths[(k + 3)%4]
					d = (a.x + (a.y - b.y)/ab*da, a.y - (a.x - b.x)/ab*da)
					nodes[(k + 3)%4].append(d)
		# finally, average the new coordinates together to place the new Nodes
		for k in range(4):
			if type(nodes[k]) is list:
				assert len(nodes[k]) > 0, nodes
				nodes[k] = Node(i_sw + k//2, (j_sw + [0, 1, 1, 0][k])%max_j,
				                *np.mean(nodes[k], axis=0))

		self.sw: Node = nodes[0]
		self.se: Node = nodes[1]
		self.ne: Node = nodes[2]
		self.nw: Node = nodes[3]
		self.all_nodes = nodes

		self.top = top
		self.bottom = bottom
		self.hite = hite


if __name__ == "__main__":
	# start by defining a grid of Cells
	ф = np.linspace(-np.pi/2, np.pi/2, int(180*RESOLUTION) + 1)
	dф = ф[1] - ф[0]
	num_ф = ф.size - 1
	λ = np.linspace(-np.pi, np.pi, int(360*RESOLUTION) + 1)
	dλ = λ[1] - λ[0]
	num_λ = λ.size - 1
	nodes = np.empty((num_ф + 1, num_λ), dtype=Node)
	cells = np.empty((num_ф, num_λ), dtype=Cell)

	# determine which of these cells belong in this section
	include = np.full(cells.shape, True)
	include[:, 6] = False

	# set a seed
	nodes[0, 0] = Node(0, 0, 0, 0)
	nodes[0, 1] = Node(0, 1, EARTH_RADIUS*dф, 0)
	# then bild out from there
	cue = [(0, 0)]
	while len(cue) > 0:
		i, j = cue.pop()
		if cells[i, j] is None and include[i, j]:
			cells[i, j] = Cell(i, j, num_λ,
			                   nodes[i, j], nodes[i, (j + 1)%num_λ],
			                   nodes[i + 1, (j + 1)%num_λ], nodes[i + 1, j],
			                   EARTH_RADIUS*dλ * (1 + np.cos(ф[i + 1])),
			                   EARTH_RADIUS*dλ * (1 + np.cos(ф[i])),
			                   EARTH_RADIUS*dф)
			for node in cells[i, j].all_nodes:
				nodes[node.i, node.j] = node
			for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
				if i + di >= 0 and i + di < num_ф:
					if cells[i + di, (j + dj)%num_λ] is None:
						cue.append((i + di, (j + dj + cells.shape[1])%num_λ))

			plt.clf()
			plt.scatter([node.x for node in nodes.ravel() if node is not None], [node.y for node in nodes.ravel() if node is not None])
			plt.axis("equal")
			plt.pause(.01)
	plt.show()