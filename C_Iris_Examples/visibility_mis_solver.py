import numpy as np
from pydrake.all import (MathematicalProgram, SolverOptions, 
			            Solve, CommonSolverOption)

from scipy.sparse.csgraph import connected_components
from scipy.sparse import lil_matrix, csr_matrix
import typing

def get_connected_components(ad_mat : lil_matrix): 
	graph = ad_mat.tocsr()
    
	# Find the connected components using scipy's connected_components function
	num_components, labels = connected_components(csgraph=graph, directed=False)

	# Initialize an empty dictionary to store the components
	components = {}

	# Iterate over the nodes and assign them to their respective components
	for node, component in enumerate(labels):
		if component not in components:
			components[component] = [node]
		else:
			components[component].append(node)

	# Convert the dictionary of components into a list of lists
	connected_components_list = list(components.values())

	return connected_components_list


def solve_max_independent_set_integer(adjacency_matrix):
	components = get_connected_components(adjacency_matrix)
	#components = [[i for i in range(adjacency_matrix.shape[0])]]
	costs = []
	vertices = []
	for comp in components:
		#reduce ad_mat to connected components
		adj_mat = adjacency_matrix[:,comp]
		adj_mat = adj_mat[comp,:]
		
		n = adj_mat.shape[0]
		

		prog = MathematicalProgram()
		v = prog.NewBinaryVariables(n)
		prog.AddLinearCost(-np.sum(v))
		for i in range(0,n):
			for j in range(i,n):
				if adj_mat[i,j]:
					prog.AddLinearConstraint(v[i] + v[j] <= 1)

		solver_options = SolverOptions()
		solver_options.SetOption(CommonSolverOption.kPrintToConsole, 1)

		result = Solve(prog, solver_options=solver_options)
		costs.append(-result.get_optimal_cost())
		vertices+=(np.array(comp)[np.nonzero(result.GetSolution(v))[0]]).tolist()

	return np.sum(costs), np.array(vertices)


if __name__ == "__main__":
    # Code to be executed when the script is run directly

	adj_matrix = lil_matrix((6, 6))
	adj_matrix[0, 1] = 1
	adj_matrix[1, 0] = 1
	adj_matrix[1, 2] = 1
	adj_matrix[2, 1] = 1
	adj_matrix[3, 4] = 1
	adj_matrix[4, 3] = 1
	adj_matrix[4, 5] = 1
	adj_matrix[5, 4] = 1

	a = get_connected_components(adj_matrix)
	cost, mis = solve_max_independent_set_integer(adj_matrix)

	print('done')