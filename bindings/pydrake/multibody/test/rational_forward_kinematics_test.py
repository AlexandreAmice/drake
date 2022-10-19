import unittest
import numpy as np
import pydrake.multibody.rational_forward_kinematics as mut
import pydrake.symbolic as sym
from pydrake.multibody.plant import (
    AddMultibodyPlantSceneGraph,
    CoulombFriction,
)
from pydrake.multibody.parsing import (
    Parser, )
from pydrake.math import (RigidTransform, RotationMatrix)
from pydrake.geometry import Box
from pydrake.common import FindResourceOrThrow
from pydrake.systems.framework import (
    DiagramBuilder, )
from pydrake.solvers import mathematicalprogram as mp


class IiwaCspaceTest(unittest.TestCase):
    def setUp(self):
        file_path = "drake/manipulation/models/iiwa_description/sdf/" +\
            "iiwa14_no_collision.sdf"
        builder = DiagramBuilder()
        self.plant, self.scene_graph = AddMultibodyPlantSceneGraph(
            builder, time_step=0.)
        parser = Parser(self.plant)
        parser.AddModelFromFile(FindResourceOrThrow(file_path))
        self.plant.WeldFrames(self.plant.world_frame(),
                              self.plant.GetFrameByName("iiwa_link_0"))
        # Now add some collision geometry.
        self.link7_box_id = self.plant.RegisterCollisionGeometry(
            self.plant.GetBodyByName("iiwa_link_7"), RigidTransform(),
            Box(0.1, 0.2, 0.3), "link7_box", CoulombFriction())
        self.link5_box_id = self.plant.RegisterCollisionGeometry(
            self.plant.GetBodyByName("iiwa_link_5"), RigidTransform(),
            Box(0.1, 0.05, 0.05), "link5_box", CoulombFriction())
        self.world_box_id = self.plant.RegisterCollisionGeometry(
            self.plant.world_body(),
            RigidTransform(RotationMatrix(), np.array([0., 0., 0.])),
            Box(0.2, 0.2, 0.3), "world_box", CoulombFriction())
        self.plant.Finalize()
        self.diagram = builder.Build()

    def test_get_collision_geometries(self):
        collision_geometries = mut.GetCollisionGeometries(
            diagram=self.diagram,
            plant=self.plant,
            scene_graph=self.scene_graph)
        self.assertEqual(len(collision_geometries), 3)

    def test_cspace_free_region_constructor(self):
        dut = mut.CspaceFreeRegion(self.diagram,
                                   self.plant,
                                   self.scene_graph,
                                   mut.SeparatingPlaneOrder.kAffine,
                                   mut.CspaceRegionType.kGenericPolytope,
                                   separating_polytope_delta=1.)
        self.assertEqual(len(dut.separating_planes()), 3)
        self.assertEqual(dut.separating_planes()[0].a.shape, (3,))

    def test_generate_tuples_for_bilinear_alternation(self):
        dut = mut.CspaceFreeRegion(self.diagram,
                                   self.plant,
                                   self.scene_graph,
                                   mut.SeparatingPlaneOrder.kAffine,
                                   mut.CspaceRegionType.kGenericPolytope,
                                   separating_polytope_delta=1.)
        q_star = np.zeros(7)
        alternation_tuples, d_minus_Ct, t_lower, t_upper, t_minus_t_lower,\
            t_upper_minus_t, C, d, lagrangian_gram_vars, verified_gram_vars,\
            separating_plane_vars, separating_plane_to_tuples,\
            separating_plane_lorentz_cone_constraints = \
            dut.GenerateTuplesForBilinearAlternation(
                q_star=q_star, filtered_collision_pairs=set(), C_rows=20)

    def construct_initial_cspace_polytope(self, dut, diagram):
        diagram_context = diagram.CreateDefaultContext()
        context = self.plant.GetMyMutableContextFromRoot(diagram_context)
        q_star = np.zeros(7)

        # Build a small c-space polytope C*t <= d around q_not_in_collision
        q_not_in_collision = np.array([0.5, 0.3, -0.3, 0.1, 0.4, 0.2, 0.1])
        self.plant.SetPositions(context, q_not_in_collision)
        self.assertFalse(dut.IsPostureInCollision(context))

        C = np.array([[1, 0, 0, 0, 2, 0, 0], [-1, 0, 0, 0, 0, 1, 0],
                      [0, 1, 1, 0, 0, 0, 1], [0, -1, -2, 0, 0, -1, 0],
                      [1, 1, 0, 2, 0, 0, 1], [1, 0, 2, -1, 0, 1, 0],
                      [0, -1, 2, -2, 1, 3, 2], [0, 1, -2, 1, 2, 4, 3],
                      [0, 3, -2, 2, 0, 1, -1], [1, 0, 3, 2, 0, -1, 1],
                      [0, 1, -1, -2, 3, -2, 1], [1, 0, -1, 1, 3, 2, 0],
                      [-1, -0.1, -0.2, 0, 0.3, 0.1, 0.1],
                      [-2, 0.1, 0.2, 0.2, -0.3, -0.1, 0.1],
                      [-1, 1, 1, 0, -1, 1, 0], [0, 0.2, 0.1, 0, -1, 0.1, 0],
                      [0.1, 2, 0.2, 0.1, -0.1, -0.2, 0.1],
                      [-0.1, -2, 0.1, 0.2, -0.15, -0.1, -0.1],
                      [0.3, 0.5, 0.1, 0.7, -0.4, 1.2, 3.1],
                      [-0.5, 0.3, 0.2, -0.5, 1.2, 0.7, -0.5],
                      [0.4, 0.6, 1.2, -0.3, -0.5, 1.2, -0.1],
                      [1.5, -0.1, 0.6, 1.5, 0.4, 2.1, 0.3],
                      [0.5, 1.5, 0.3, 0.2, 1.5, -0.1, 0.5],
                      [0.5, 0.2, -0.1, 1.2, -0.3, 1.1, -0.4]])
        # Now I normalize each row of C. Because later when we search for the
        # polytope we have the constraint that |C.row()|<=1, so it is better to
        # start with a C satisfying this constraint.
        C_row_norm = np.linalg.norm(C, ord=2, axis=1)
        C = C / C_row_norm[:, np.newaxis]
        # Now I take some samples of t slightly away from q_not_in_collision.
        # C * t <= d contains all these samples.

        t_samples = np.empty((7, 6))
        t_samples[:, 0] = np.tan((q_not_in_collision - q_star) / 2)
        t_samples[:, 1] = t_samples[:, 0] + np.array(
            [0.11, -0.02, 0.03, 0.01, 0, 0.02, 0.02])
        t_samples[:, 2] = t_samples[:, 0] + np.array(
            [-0.005, 0.01, -0.02, 0.01, 0.005, 0.01, -0.02])
        t_samples[:, 3] = t_samples[:, 0] + np.array(
            [0.02, -0.13, 0.01, 0.02, -0.03, 0.01, 0.15])
        t_samples[:, 4] = t_samples[:, 0] + np.array(
            [0.01, -0.04, 0.003, 0.01, -0.01, -0.11, -0.08])
        t_samples[:, 5] = t_samples[:, 0] + np.array(
            [-0.01, -0.02, 0.013, -0.02, 0.03, -0.03, -0.1])
        d = np.max(C @ t_samples, axis=1)

        return q_star, C, d

    def test_construct_lagrangian_program(self):
        dut = mut.CspaceFreeRegion(self.diagram,
                                   self.plant,
                                   self.scene_graph,
                                   mut.SeparatingPlaneOrder.kAffine,
                                   mut.CspaceRegionType.kGenericPolytope,
                                   separating_polytope_delta=1.)
        q_star = np.zeros(7)
        alternation_tuples, d_minus_Ct, t_lower, t_upper, t_minus_t_lower,\
            t_upper_minus_t, C_var, d_var, lagrangian_gram_vars,\
            verified_gram_vars, separating_plane_vars, \
            separating_plane_to_tuples, \
            separating_plane_lorentz_cone_constraints =\
            dut.GenerateTuplesForBilinearAlternation(
                q_star=q_star, filtered_collision_pairs=set(), C_rows=24)

        q_star, C, d = self.construct_initial_cspace_polytope(
            dut, self.diagram)
        P = np.empty((7, 7), dtype=sym.Variable)
        q = np.empty(7, dtype=sym.Variable)
        verification_option = mut.VerificationOption()
        redundant_tighten = 0.5
        # Join separating_plane_lorentz_cone_constraints to a single list
        # from a list of sublists.
        separating_plane_lorentz_cone_constraints = sum(
            separating_plane_lorentz_cone_constraints, [])
        prog_lagrangian = dut.ConstructLagrangianProgram(
            alternation_tuples, C, d, lagrangian_gram_vars, verified_gram_vars,
            separating_plane_vars, separating_plane_lorentz_cone_constraints,
            t_lower, t_upper, verification_option, redundant_tighten)
        P, q = mut.AddInscribedEllipsoid(prog_lagrangian, C, d, t_lower,
                                         t_upper)
        result_lagrangian = mp.Solve(prog_lagrangian)
        self.assertTrue(result_lagrangian.is_success())

        lagrangian_gram_var_vals = result_lagrangian.GetSolution(
            lagrangian_gram_vars)
        P_sol = result_lagrangian.GetSolution(P)
        q_sol = result_lagrangian.GetSolution(q)

        prog_polytope = dut.ConstructPolytopeProgram(
            alternation_tuples, C_var, d_var, d_minus_Ct,
            lagrangian_gram_var_vals, verified_gram_vars,
            separating_plane_vars, separating_plane_lorentz_cone_constraints,
            t_minus_t_lower, t_upper_minus_t, verification_option)
        result_polytope = mp.Solve(prog_polytope)
        self.assertTrue(result_polytope.is_success())

    def test_cspace_polytope_bilinear_alternation(self):
        dut = mut.CspaceFreeRegion(self.diagram,
                                   self.plant,
                                   self.scene_graph,
                                   mut.SeparatingPlaneOrder.kAffine,
                                   mut.CspaceRegionType.kGenericPolytope,
                                   separating_polytope_delta=1.)
        q_star, C_init, d_init = self.construct_initial_cspace_polytope(
            dut, self.diagram)
        filtered_collision_pairs = set()
        bilinear_alternation_option = mut.BilinearAlternationOption()
        bilinear_alternation_option.max_iters = 2
        bilinear_alternation_option.lagrangian_backoff_scale = 0.01
        bilinear_alternation_option.polytope_backoff_scale = 0.05
        solver_options = mp.SolverOptions()
        cspace_free_region_solution, polytope_volumes, \
            ellipsoid_determinants = \
                dut.CspacePolytopeBilinearAlternation(
                    q_star, filtered_collision_pairs, C_init, d_init,
                    bilinear_alternation_option, solver_options,
                    t_inner_pts=None, inner_polytope=None)

    def test_cspace_polytope_binary_search(self):
        dut = mut.CspaceFreeRegion(self.diagram,
                                   self.plant,
                                   self.scene_graph,
                                   mut.SeparatingPlaneOrder.kAffine,
                                   mut.CspaceRegionType.kGenericPolytope,
                                   separating_polytope_delta=1.)
        q_star, C_init, d_init = self.construct_initial_cspace_polytope(
            dut, self.diagram)
        filtered_collision_pairs = set()
        binary_search_option = mut.BinarySearchOption()
        binary_search_option.epsilon_max = 1
        binary_search_option.epsilon_min = 0.
        binary_search_option.max_iters = 3
        solver_options = mp.SolverOptions()
        cspace_free_region_solution = dut.CspacePolytopeBinarySearch(
            q_star, filtered_collision_pairs, C_init, d_init,
            binary_search_option, solver_options, None, None)
        np.testing.assert_array_less(d_init, cspace_free_region_solution.d)
