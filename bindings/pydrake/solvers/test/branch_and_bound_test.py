import unittest

import numpy as np

from pydrake.solvers import (
    MathematicalProgram,
    MixedIntegerBranchAndBound,
    OsqpSolver,
    SolutionResult,
)

from functools import partial

import time

class TestMixedIntegerBranchAndBound(unittest.TestCase):
    def test(self):
        prog = MathematicalProgram()
        x = prog.NewContinuousVariables(2)
        b = prog.NewBinaryVariables(2)

        prog.AddLinearConstraint(x[0] + 2 * x[1] + b[0] == 2.)
        prog.AddLinearConstraint(x[0] - 3.1 * b[1] >= 1)
        prog.AddLinearConstraint(b[1] + 1.2 * x[1] - b[0] <= 5)
        prog.AddQuadraticCost(x[0] * x[0])

        dut = MixedIntegerBranchAndBound(prog, OsqpSolver().solver_id())
        def check_solution(bnb):
            solution_result = bnb.Solve()
            self.assertEqual(solution_result, SolutionResult.kSolutionFound)
            self.assertAlmostEqual(bnb.GetOptimalCost(), 1.)
            self.assertAlmostEqual(bnb.GetSubOptimalCost(0), 1.)
            self.assertAlmostEqual(bnb.GetSolution(x[0], 0), 1.)
            self.assertAlmostEqual(bnb.GetSolution(x[0], 1), 1.)
            np.testing.assert_allclose(bnb.GetSolution(x, 0), [1., 0.], atol=1e-12)
            print("done check solution")
            time.sleep(1)

        dut = MixedIntegerBranchAndBound(prog, OsqpSolver().solver_id())
        dut.SetNodeSelectionMethod(
            MixedIntegerBranchAndBound.NodeSelectionMethod.kDepthFirst)
        check_solution(dut)

        dut = MixedIntegerBranchAndBound(prog, OsqpSolver().solver_id())
        dut.SetNodeSelectionMethod(
            MixedIntegerBranchAndBound.NodeSelectionMethod.kMinLowerBound)
        check_solution(dut)

        dut = MixedIntegerBranchAndBound(prog, OsqpSolver().solver_id())
        dut.SetNodeSelectionMethod(
            MixedIntegerBranchAndBound.NodeSelectionMethod.kUserDefined)

        # Python implementation of LeftMostNodeInSubTree from
        # branch_and_bound_test.cc
        def LeftMostNodeInSubTree(branch_and_bound_subtree_root, branch_and_bound):
            if branch_and_bound_subtree_root.isLeaf():
                if branch_and_bound.IsLeafNodeFathomed(branch_and_bound_subtree_root):
                    return None
                else:
                    return branch_and_bound_subtree_root
            else:
                left_most_node_left_tree = LeftMostNodeInSubTree(branch_and_bound_subtree_root.left_child(),
                                                                 branch_and_bound)
                if not left_most_node_left_tree:
                    return LeftMostNodeInSubTree(branch_and_bound_subtree_root.right_child(),
                                                 branch_and_bound)
                return left_most_node_left_tree
        dut.SetNodeSelectionMethod(partial(LeftMostNodeInSubTree, dut.root()))
        check_solution(dut)

