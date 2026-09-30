import copy
import unittest

import numpy as np

from pydrake.common.test_utilities import numpy_compare
from pydrake.solvers import (
    MathematicalProgram,
    ScsSolver,
    SolverCacheStatus,
    SolverOptions,
    SolverType,
)


class TestScsSolver(unittest.TestCase):
    def test_scs_solver(self):
        prog = MathematicalProgram()
        x = prog.NewContinuousVariables(2, "x")
        prog.AddLinearConstraint(x[0] >= 1)
        prog.AddLinearConstraint(x[1] >= 1)
        prog.AddQuadraticCost(np.eye(2), np.zeros(2), x)
        solver = ScsSolver()
        self.assertEqual(solver.solver_id(), ScsSolver.id())

        self.assertTrue(solver.available())
        self.assertEqual(solver.solver_id().name(), "SCS")
        self.assertEqual(solver.SolverName(), "SCS")
        self.assertEqual(solver.solver_type(), SolverType.kScs)

        result = solver.Solve(prog, None, None)
        self.assertTrue(result.is_success())
        # Use a loose tolerance for the check since the correctness of the
        # solver has been validated in the C++ code, while the Python test
        # just verifies if the binding works or not.
        atol = 1e-3
        numpy_compare.assert_float_allclose(
            result.GetSolution(x), [1.0, 1.0], atol=atol
        )
        numpy_compare.assert_float_allclose(
            result.get_solver_details().primal_objective, 1.0, atol=atol
        )
        numpy_compare.assert_float_allclose(
            result.get_solver_details().primal_residue,
            np.array([0.0]),
            atol=atol,
        )
        numpy_compare.assert_float_allclose(
            result.get_solver_details().y, np.array([1.0, 1.0]), atol=atol
        )

    def test_repeated_solve_cache(self):
        solver = ScsSolver()
        prog = MathematicalProgram()
        x = prog.NewContinuousVariables(2)
        prog.AddQuadraticCost(np.eye(2), np.zeros(2), x)
        bounds = prog.AddBoundingBoxConstraint(1.0, 3.0, x)
        options = SolverOptions()
        options.SetOption(solver.id(), "retain_solver_cache", 1)
        result = solver.Solve(prog, None, options)
        self.assertTrue(result.has_solver_cache())
        self.assertEqual(
            result.get_solver_details().cache.status, SolverCacheStatus.kCreated
        )
        snapshots = [copy.copy(result), copy.deepcopy(result)]
        for snapshot in snapshots:
            self.assertFalse(snapshot.has_solver_cache())
        bounds.evaluator().UpdateLowerBound(np.full(2, 2.0))
        solver.Solve(prog, None, options, result)
        self.assertTrue(result.is_success())
        self.assertEqual(
            result.get_solver_details().cache.status, SolverCacheStatus.kUpdated
        )
        np.testing.assert_allclose(result.GetSolution(x), [2.0, 2.0], atol=1e-3)
        for snapshot in snapshots:
            np.testing.assert_allclose(
                snapshot.GetSolution(x), [1.0, 1.0], atol=1e-3
            )
        solver.Solve(prog, None, options, result)
        self.assertEqual(
            result.get_solver_details().cache.status, SolverCacheStatus.kReused
        )
        with self.assertRaisesRegex(
            ValueError, "different MathematicalProgram"
        ):
            solver.Solve(prog.Clone(), None, options, result)
        options.SetOption(solver.id(), "retain_solver_cache", 0)
        solver.Solve(prog, None, options, result)
        self.assertFalse(result.has_solver_cache())

    def unavailable(self):
        """Per the BUILD file, this test is only run when SCS is disabled."""
        solver = ScsSolver()
        self.assertFalse(solver.available())
