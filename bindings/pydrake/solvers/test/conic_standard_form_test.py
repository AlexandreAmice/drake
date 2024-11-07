import numpy as np
import unittest

from pydrake.solvers import (
    ConicStandardForm,
    MathematicalProgram,
    ProgramAttribute
)
from pydrake.symbolic import Variables


class TestConicStandardForm(unittest.TestCase):
    def test_TestConicStandardForm(self):
        prog = MathematicalProgram()
        y = prog.NewContinuousVariables(2, "y")
        A = np.array([[0.5, 0.7], [-.2, 0.4], [-2.3, -4.5]])
        b = np.array([1.3, -.24, 0.25])
        prog.AddLinearConstraint(A, b, np.inf*np.ones(3), y)
        c = np.array([0.1, 0.2])
        d = 0.5
        prog.AddLinearCost(c, d, y)
        conic_standard_form = ConicStandardForm(prog=prog)
        np.testing.assert_array_equal(conic_standard_form.A().toarray(), A)
        np.testing.assert_array_equal(
            conic_standard_form.b().toarray().flatten(), -b)
        np.testing.assert_array_equal(
            conic_standard_form.c().toarray().flatten(), c)
        self.assertEqual(conic_standard_form.d(), d)

        self.assertTrue(ProgramAttribute.kLinearConstraint in
                        conic_standard_form.
                        attributes_to_start_end_pairs().keys())

        standard_form_prog = conic_standard_form.MakeProgram()
        self.assertEqual(len(standard_form_prog.linear_constraints()), 1)
        self.assertEqual(len(standard_form_prog.linear_costs()), 1)
