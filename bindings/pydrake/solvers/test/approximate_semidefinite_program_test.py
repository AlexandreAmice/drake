import numpy as np
import unittest

from pydrake.solvers import (
    MakeDiagonallyDominantInnerApproximation,
    MakeScaledDiagonallyDominantInnerApproximation,
    MakeDiagonallyDominantDualConeOuterApproximation,
    MakeScaledDiagonallyDominantDualConeOuterApproximation,
    MathematicalProgram,
)


class TestApproximateSemidefiniteProgram(unittest.TestCase):

    def make_test_program(self):
        # A test program with many types of constraints, and one PSD
        # constraint.
        prog = MathematicalProgram()
        x = prog.NewContinuousVariables(3)
        X = prog.NewSymmetricContinuousVariables(3)

        lin_con = prog.AddLinearConstraint(x[0] + x[1] <= 1)

        lin_eq_con = prog.AddLinearEqualityConstraint(x[0] + x[1] == 1)

        bb_con = prog.AddBoundingBoxConstraint(0, 1, x)

        lorentz_con = prog.AddLorentzConeConstraint(x)

        rotated_lorentz_con = prog.AddRotatedLorentzConeConstraint(x)

        psd_con = prog.AddPositiveSemidefiniteConstraint(X)

        lmi_con = prog.AddLinearMatrixInequalityConstraint(
            [np.eye(3), np.eye(3), 2 * np.ones((3, 3))], x[:2])

        exponential_con = prog.AddExponentialConeConstraint(
            A=np.array([[1, 3], [2, 4], [0, 1]]), b=np.array([0, 1, 3]),
            vars=x[:2])
        lcp_con = prog.AddLinearComplementarityConstraint(
            np.eye(3), np.ones((3,)), x)
        return prog

    def test_MakeDiagonallyDominantInnerApproximation(self):
        prog = self.make_test_program()
        clone = prog.Clone()

        MakeDiagonallyDominantInnerApproximation(clone)
        # Ensures that we haven't mutated prog. This is important to check due
        # to the difference in pointer semantics.
        self.assertEqual(len(prog.positive_semidefinite_constraints()), 1)
        self.assertEqual(len(clone.positive_semidefinite_constraints()), 0)

        MakeDiagonallyDominantInnerApproximation(prog)
        self.assertEqual(len(prog.positive_semidefinite_constraints()), 0)

    def test_MakeScaledDiagonallyDominantInnerApproximation(self):
        prog = self.make_test_program()
        clone = prog.Clone()

        MakeScaledDiagonallyDominantInnerApproximation(clone)
        # Ensures that we haven't mutated prog. This is important to check due
        # to the difference in pointer semantics.
        self.assertEqual(len(prog.positive_semidefinite_constraints()), 1)
        self.assertEqual(len(clone.positive_semidefinite_constraints()), 0)

        MakeScaledDiagonallyDominantInnerApproximation(prog)
        self.assertEqual(len(prog.positive_semidefinite_constraints()), 0)

    def test_MakeDiagonallyDominantDualConeOuterApproximation(self):
        prog = self.make_test_program()
        clone = prog.Clone()

        MakeDiagonallyDominantDualConeOuterApproximation(clone)
        # Ensures that we haven't mutated prog. This is important to check due
        # to the difference in pointer semantics.
        self.assertEqual(len(prog.positive_semidefinite_constraints()), 1)
        self.assertEqual(len(clone.positive_semidefinite_constraints()), 0)

        MakeDiagonallyDominantDualConeOuterApproximation(prog)
        self.assertEqual(len(prog.positive_semidefinite_constraints()), 0)

    def test_MakeScaledDiagonallyDominantDualConeOuterApproximation(self):
        prog = self.make_test_program()
        clone = prog.Clone()

        MakeScaledDiagonallyDominantDualConeOuterApproximation(clone)
        # Ensures that we haven't mutated prog. This is important to check due
        # to the difference in pointer semantics.
        self.assertEqual(len(prog.positive_semidefinite_constraints()), 1)
        self.assertEqual(len(clone.positive_semidefinite_constraints()), 0)

        MakeScaledDiagonallyDominantDualConeOuterApproximation(prog)
        self.assertEqual(len(prog.positive_semidefinite_constraints()), 0)
