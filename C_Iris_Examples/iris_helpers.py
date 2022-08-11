import numpy as np

from pydrake.geometry.optimization import HPolyhedron
from pydrake.solvers.clp import ClpSolver
from pydrake.solvers.mathematicalprogram import MathematicalProgram
from pydrake.solvers.mosek import MosekSolver

def simplify_hpoly(hpoly):
    new_A = np.empty_like(hpoly.A())
    new_b = np.empty_like(hpoly.b())
    nq = hpoly.ambient_dimension()
    new_A[:2*nq] = hpoly.A()[:2*nq]
    new_b[:2*nq] = hpoly.b()[:2*nq]
    hyperplanes = 2*nq
    
    solver = ClpSolver()
#     solver = MosekSolver()

    prog = MathematicalProgram()
    q = prog.NewContinuousVariables(hpoly.ambient_dimension(), "q")
    hpoly.AddPointInSetConstraints(prog, q)
    hyperplane_cost = prog.AddLinearCost(-hpoly.A()[0], hpoly.b()[0], q)
    for ii in range(2*nq, hpoly.A().shape[0]):
        hyperplane_cost.evaluator().UpdateCoefficients(-hpoly.A()[ii], hpoly.b()[ii])
        result = solver.Solve(prog)
        if result.get_optimal_cost() < 1e-8:
            new_A[hyperplanes] = hpoly.A()[ii]
            new_b[hyperplanes] = hpoly.b()[ii]
            hyperplanes += 1
#         else:
#             print(result.get_optimal_cost(), ",", end="")
            
    return HPolyhedron(new_A[:hyperplanes], new_b[:hyperplanes])

def sorted_vertices(vpoly):
    assert vpoly.ambient_dimension() == 2
    poly_center = np.sum(vpoly.vertices(), axis=1) / vpoly.vertices().shape[1]
    vertex_vectors = vpoly.vertices() - np.expand_dims(poly_center, 1)
    sorted_index = np.arctan2(vertex_vectors[1], vertex_vectors[0]).argsort()
    return vpoly.vertices()[:, sorted_index]
