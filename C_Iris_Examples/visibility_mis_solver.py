import numpy as np
from pydrake.all import (MathematicalProgram, SolverOptions, 
			            Solve, CommonSolverOption)


def solve_max_independent_set_integer(adj_mat):
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
	return -result.get_optimal_cost(), np.nonzero(result.GetSolution(v))[0]