import unittest

import numpy as np

from pydrake.solvers import (
    ConicStandardForm,
    ConicStandardFormOptions,
    MathematicalProgram,
    ProgramAttribute,
)


class TestConicStandardForm(unittest.TestCase):
    def test_TestConicStandardForm(self):
        prog = MathematicalProgram()
        y = prog.NewContinuousVariables(2, "y")
        A = np.array([[0.5, 0.7], [-0.2, 0.4], [-2.3, -4.5]])
        b = np.array([1.3, -0.24, 0.25])
        prog.AddLinearConstraint(A, b, np.inf * np.ones(3), y)
        c = np.array([0.1, 0.2])
        d = 0.5
        prog.AddLinearCost(c, d, y)
        conic_standard_form = ConicStandardForm(prog=prog)
        np.testing.assert_array_equal(conic_standard_form.A().toarray(), A)
        np.testing.assert_array_equal(
            conic_standard_form.b().toarray().flatten(), -b
        )
        np.testing.assert_array_equal(
            conic_standard_form.c().toarray().flatten(), c
        )
        self.assertEqual(conic_standard_form.d(), d)

        self.assertTrue(
            ProgramAttribute.kLinearConstraint
            in conic_standard_form.attributes_to_start_end_pairs().keys()
        )

        standard_form_prog = conic_standard_form.MakeProgram()
        self.assertEqual(len(standard_form_prog.linear_constraints()), 1)
        self.assertEqual(len(standard_form_prog.linear_costs()), 1)

    def test_conic_standard_form_options(self):
        opts = ConicStandardFormOptions()
        self.assertFalse(opts.keep_quadratic_costs)
        self.assertTrue(opts.parse_bounding_box_constraints_as_positive_orthant)

        opts.keep_quadratic_costs = True
        opts.parse_bounding_box_constraints_as_positive_orthant = False

        prog = MathematicalProgram()
        x = prog.NewContinuousVariables(1, "x")
        prog.AddQuadraticCost(2 * np.eye(1), np.zeros(1), x)
        prog.AddBoundingBoxConstraint(0, 1, x)

        conic_standard_form = ConicStandardForm(prog=prog, options=opts)
        attrs = conic_standard_form.attributes_to_start_end_pairs()
        self.assertEqual(len(attrs[ProgramAttribute.kLinearConstraint]), 2)

        standard_form_prog = conic_standard_form.MakeProgram()
        self.assertEqual(len(standard_form_prog.quadratic_costs()), 1)
