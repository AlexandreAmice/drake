import copy
import unittest

import numpy as np

from pydrake.solvers import (
    MathematicalProgram,
    OsqpSolver,
    SolverCacheStatus,
    SolverOptions,
    SolverType,
)


class TestOsqpSolver(unittest.TestCase):
    def test_osqp_solver(self):
        prog = MathematicalProgram()
        x = prog.NewContinuousVariables(2, "x")
        constraint1 = prog.AddLinearConstraint(x[0] >= 1)
        constraint2 = prog.AddLinearConstraint(x[1] >= 1)
        prog.AddQuadraticCost(np.eye(2), np.zeros(2), x)
        solver = OsqpSolver()
        self.assertEqual(solver.solver_id(), OsqpSolver.id())
        self.assertTrue(solver.available())
        self.assertEqual(solver.solver_type(), SolverType.kOsqp)
        result = solver.Solve(prog, None, None)
        self.assertTrue(result.is_success())
        x_expected = np.array([1, 1])
        self.assertTrue(np.allclose(result.GetSolution(x), x_expected))
        self.assertEqual(result.get_solver_details().status_val, 1)
        self.assertEqual(result.get_solver_details().primal_res, 0.0)
        np.testing.assert_allclose(
            result.get_solver_details().y, np.array([-1.0, -1.0])
        )
        np.testing.assert_allclose(result.GetDualSolution(constraint1), [1.0])
        np.testing.assert_allclose(result.GetDualSolution(constraint2), [1.0])

    def test_repeated_solve_cache(self):
        solver = OsqpSolver()
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
        """Per the BUILD file, this test is only run when OSQP is disabled."""
        solver = OsqpSolver()
        self.assertFalse(solver.available())
