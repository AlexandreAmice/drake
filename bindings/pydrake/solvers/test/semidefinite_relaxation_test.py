import numpy as np
import unittest

from pydrake.solvers import (
    MakeSemidefiniteRelaxation,
    MathematicalProgram,
)
from pydrake.symbolic import Variables


class TestSemidefiniteRelaxation(unittest.TestCase):
    def test_MakeSemidefiniteRelaxation(self):
        prog = MathematicalProgram()
        y = prog.NewContinuousVariables(2, "y")
        A = np.array([[0.5, 0.7], [-.2, 0.4], [-2.3, -4.5]])
        lb = np.array([1.3, -.24, 0.25])
        ub = np.array([5.6, 0.1, 1.4])
        prog.AddLinearConstraint(A, lb, ub, y)
        relaxation = MakeSemidefiniteRelaxation(prog)

        self.assertEqual(relaxation.num_vars(), 6)
        self.assertEqual(len(relaxation.positive_semidefinite_constraints()),
                         1)
        self.assertEqual(len(relaxation.bounding_box_constraints()), 1)
        self.assertEqual(len(relaxation.linear_constraints()), 2)


    def test_MakeSemidefiniteRelaxation(self):
        prog = MathematicalProgram()
        y = prog.NewContinuousVariables(2, "y")
        A = np.array([[0.5, 0.7], [-.2, 0.4], [-2.3, -4.5]])
        lb = np.array([1.3, -.24, 0.25])
        ub = np.array([5.6, 0.1, 1.4])
        prog.AddLinearConstraint(A, lb, ub, y)

        sparsity = {Variables([y[0]]): True,
                    Variables([y[1]]): False}

        relaxation = MakeSemidefiniteRelaxation(prog, sparsity)

        self.assertEqual(relaxation.num_vars(), 6)
        self.assertEqual(len(relaxation.positive_semidefinite_constraints()),
                         2)
        self.assertEqual(len(relaxation.bounding_box_constraints()), 1)
        self.assertEqual(len(relaxation.linear_constraints()), 2)
