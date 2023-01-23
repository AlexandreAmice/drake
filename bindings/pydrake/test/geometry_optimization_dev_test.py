import pydrake.geometry.optimization_dev as mut

import unittest

import numpy as np

from pydrake.common import FindResourceOrThrow
from pydrake.geometry import Capsule, Sphere, Cylinder, Box, Convex
from pydrake.geometry import ProximityProperties, GeometryId
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import MultibodyPlant, AddMultibodyPlantSceneGraph
from pydrake.systems.framework import DiagramBuilder
from pydrake.solvers import MosekSolver, CommonSolverOption, SolverOptions
from pydrake.math import RigidTransform
from pydrake.solvers import MosekSolver, ScsSolver
from pydrake.symbolic import Polynomial
from pydrake.multibody.rational import RationalForwardKinematics


class TestGeometeryOptimizationDev(unittest.TestCase):
    def setUp(self):
        unittest.TestCase.setUp(self)
        limits_urdf = """
                <robot name="limits">
                  <link name="movable">
                    <collision>
                      <geometry><box size="0.1 0.1 0.1"/></geometry>
                    </collision>
                     <geometry>
                         <cylinder length="0.1" radius="0.2"/>
                    </geometry>
                    <geometry>
                         <capsule length="0.1" radius="0.2"/>
                    </geometry>
                    <geometry>
                         <sphere radius="0.2"/>
                    </geometry>
                  </link>
                  <link name="unmovable">
                    <collision>
                      <geometry><box size="1 1 1"/></geometry>
                    </collision>
                  </link>
                  <joint name="movable" type="prismatic">
                    <axis xyz="1 0 0"/>
                    <limit lower="-2" upper="2"/>
                    <parent link="world"/>
                    <child link="movable"/>
                  </joint>
                  <joint name="unmovable" type = "fixed">
                        <parent link="world"/>
                        <child link="unmovable"/>
                        <origin xyz="1 0 0"/>
                  </joint>
                </robot>"""

        builder = DiagramBuilder()
        self.plant, self.scene_graph = AddMultibodyPlantSceneGraph(
            builder, MultibodyPlant(time_step=0.01))
        Parser(self.plant).AddModelsFromString(limits_urdf, "urdf")

        self.plant.Finalize()

        diagram = builder.Build()
        # test constructor
        self.cspace_free_polytope = mut.CspaceFreePolytope(
            plant=self.plant,
            scene_graph=self.scene_graph,
            plane_order=mut.SeparatingPlaneOrder.kAffine,
            q_star=np.zeros(1))

    def test_CollisionGeometry(self):
        collision_geometries = mut.GetCollisionGeometries(
            plant=self.plant, scene_graph=self.scene_graph)

        geom_type_possible_values = [
            mut.GeometryType.kPolytope,
            mut.GeometryType.kSphere,
            mut.GeometryType.kCylinder,
            mut.GeometryType.kCapsule]
        geom_shape_possible_values = [
            Capsule, Sphere, Cylinder, Box, Convex
        ]

        for geom_lst in collision_geometries.values():
            for geom in geom_lst:
                self.assertIn(geom.type(), geom_type_possible_values)
                self.assertIn(
                    type(
                        geom.geometry()),
                    geom_shape_possible_values)
                self.assertIn(geom.body_index(), collision_geometries.keys())
                self.assertGreater(geom.num_rationals(), 0)
                self.assertIsInstance(geom.X_BG(), RigidTransform)
                self.assertIsInstance(geom.id(), GeometryId)

        # Check that the plane sides are properly enumerated.
        plane_side_possible_values = [mut.PlaneSide.kPositive,
                                      mut.PlaneSide.kNegative]

    def test_CspaceFreePolytope_Options(self):
        solver_options = SolverOptions()
        solver_options.SetOption(CommonSolverOption.kPrintToConsole, 1)

        polytope_options = mut.FindPolytopeGivenLagrangianOptions()
        self.assertIsNone(polytope_options.backoff_scale)
        self.assertEqual(
            polytope_options.ellipsoid_margin_epsilon, 1e-5)
        self.assertEqual(
            polytope_options.solver_id,
            MosekSolver.id())
        self.assertIsNone(polytope_options.solver_options)
        self.assertIsNone(polytope_options.s_inner_pts)
        self.assertTrue(
            polytope_options.search_s_bounds_lagrangians)
        self.assertEqual(
            polytope_options.ellipsoid_margin_cost,
            mut.EllipsoidMarginCost.kGeometricMean)

        polytope_options.backoff_scale = 1e-3
        polytope_options.ellipsoid_margin_epsilon = 1e-6
        polytope_options.solver_id = ScsSolver.id()
        polytope_options.solver_options = solver_options
        polytope_options.s_inner_pts = np.zeros((2, 1))
        polytope_options.search_s_bounds_lagrangians = False
        polytope_options.ellipsoid_margin_cost = mut.EllipsoidMarginCost.kSum
        self.assertEqual(
            polytope_options.backoff_scale, 1e-3)
        self.assertEqual(
            polytope_options.ellipsoid_margin_epsilon, 1e-6)
        self.assertEqual(
            polytope_options.solver_id,
            ScsSolver.id())
        self.assertEqual(
            polytope_options.solver_options.common_solver_options()[
                CommonSolverOption.kPrintToConsole], 1)
        np.testing.assert_array_almost_equal(
            polytope_options.s_inner_pts, np.zeros(
                (2, 1)), 1e-5)
        self.assertFalse(
            polytope_options.search_s_bounds_lagrangians)
        self.assertEqual(
            polytope_options.ellipsoid_margin_cost,
            mut.EllipsoidMarginCost.kSum)

        lagrangian_options = \
            mut.FindSeparationCertificateGivenPolytopeOptions()
        self.assertEqual(
            lagrangian_options.num_threads, -1)
        self.assertFalse(
            lagrangian_options.verbose)
        self.assertEqual(
            lagrangian_options.solver_id,
            MosekSolver.id())
        self.assertTrue(
            lagrangian_options.terminate_at_failure)
        self.assertIsNone(
            lagrangian_options.solver_options)
        self.assertFalse(
            lagrangian_options.ignore_redundant_C)

        num_threads = 1
        lagrangian_options.num_threads = num_threads
        lagrangian_options.verbose = True
        lagrangian_options.solver_id = ScsSolver.id()
        lagrangian_options.terminate_at_failure = False
        lagrangian_options.solver_options = solver_options
        lagrangian_options.ignore_redundant_C = True
        self.assertEqual(
            lagrangian_options.num_threads,
            num_threads)
        self.assertTrue(
            lagrangian_options.verbose)
        self.assertEqual(
            lagrangian_options.solver_id,
            ScsSolver.id())
        self.assertFalse(
            lagrangian_options.terminate_at_failure)
        self.assertEqual(
            lagrangian_options.solver_options.common_solver_options()[
                CommonSolverOption.kPrintToConsole], 1)
        self.assertTrue(
            lagrangian_options.ignore_redundant_C)

        bilinear_alternation_options = mut.BilinearAlternationOptions()
        self.assertEqual(bilinear_alternation_options.max_iter, 10)
        self.assertAlmostEqual(bilinear_alternation_options.convergence_tol,
                               1e-3, 1e-10)
        self.assertAlmostEqual(bilinear_alternation_options.ellipsoid_scaling,
                               0.99, 1e-10)
        self.assertTrue(bilinear_alternation_options.
                        find_polytope_options.search_s_bounds_lagrangians)
        self.assertFalse(
            bilinear_alternation_options.find_lagrangian_options.verbose)

        bilinear_alternation_options.max_iter = 4
        bilinear_alternation_options.convergence_tol = 1e-2
        bilinear_alternation_options.find_polytope_options = polytope_options
        bilinear_alternation_options.find_lagrangian_options =\
            lagrangian_options
        bilinear_alternation_options.ellipsoid_scaling = 0.5
        self.assertEqual(bilinear_alternation_options.max_iter, 4)
        self.assertAlmostEqual(bilinear_alternation_options.convergence_tol,
                               1e-2, 1e-10)
        self.assertAlmostEqual(bilinear_alternation_options.ellipsoid_scaling,
                               0.5, 1e-10)
        self.assertFalse(bilinear_alternation_options.
                         find_polytope_options.search_s_bounds_lagrangians)
        self.assertTrue(bilinear_alternation_options.
                        find_lagrangian_options.verbose)

        binary_search_options = mut.BinarySearchOptions()
        self.assertAlmostEqual(binary_search_options.scale_max, 1, 1e-10)
        self.assertAlmostEqual(binary_search_options.scale_min, 0.01, 1e-10)
        self.assertEqual(binary_search_options.max_iter, 10)
        self.assertAlmostEqual(
            binary_search_options.convergence_tol, 1e-3, 1e-10)
        self.assertFalse(
            binary_search_options.find_lagrangian_options.verbose)

        binary_search_options.scale_max = 2
        binary_search_options.scale_min = 1
        binary_search_options.max_iter = 2
        binary_search_options.convergence_tol = 1e-5
        binary_search_options.find_lagrangian_options = lagrangian_options
        self.assertAlmostEqual(binary_search_options.scale_max, 2, 1e-10)
        self.assertAlmostEqual(binary_search_options.scale_min, 1, 1e-10)
        self.assertEqual(binary_search_options.max_iter, 2)
        self.assertAlmostEqual(
            binary_search_options.convergence_tol, 1e-5, 1e-10)
        self.assertTrue(
            binary_search_options.find_lagrangian_options.verbose)

        options = mut.Options()
        self.assertFalse(options.with_cross_y)
        options.with_cross_y = True
        self.assertTrue(options.with_cross_y)

    def test_CspaceFreePolytope_constructor_and_getters(self):
        dut = self.cspace_free_polytope
        rat_forward = dut.rational_forward_kin()
        self.assertEqual(
            rat_forward.ComputeSValue(
                np.zeros(self.plant.num_positions()),
                np.zeros(self.plant.num_positions())),
            np.zeros(self.plant.num_positions()))
        # TODO (AlexandreAmice) uncomment once I get this binding working.
        # self.assertGreaterEqual(
        #     len(dut.map_geometries_to_separating_planes().keys()), 1)
        # pair = dut.sorted_pair_method()
        self.assertGreaterEqual(
            len(dut.separating_planes()), 1)
        self.assertEqual(len(dut.y_slack()), 3)

    def test_separating_plane(self):
        # Check that the plane orders are properly enumerated.
        possible_orders = [mut.SeparatingPlaneOrder.kAffine]

        plane = self.cspace_free_polytope.separating_planes()[0]
        self.assertEqual(len(plane.a), 3)
        self.assertIsInstance(plane.b, Polynomial)
        self.assertIsInstance(
            plane.positive_side_geometry,
            mut.CollisionGeometry)
        self.assertIsInstance(
            plane.negative_side_geometry,
            mut.CollisionGeometry)
        self.assertTrue(plane.expressed_body.is_valid())
        self.assertIn(plane.plane_order, possible_orders)
        self.assertGreaterEqual(len(plane.decision_variables), 4)

    def test_CspaceFreePolytopeSearchMethods(self):

        C_init = np.vstack([np.atleast_2d(np.eye(self.plant.num_positions(
        ))), -np.atleast_2d(np.eye(self.plant.num_positions()))])
        lim = 3
        d_init = lim * np.ones((C_init.shape[0], 1))

        bilinear_alternation_options = mut.BilinearAlternationOptions()
        binary_search_options = mut.BinarySearchOptions()
        binary_search_options.scale_min = 1e-4
        bilinear_alternation_options.find_lagrangian_options.verbose = False
        binary_search_options.find_lagrangian_options.verbose = False

        result = self.cspace_free_polytope.BinarySearch(
            ignored_collision_pairs=set(),
            C=C_init,
            d=d_init,
            s_center=np.zeros(self.plant.num_positions()),
            options=binary_search_options
        )
        # Accesses all members of SearchResult
        self.assertGreaterEqual(result.num_iter, 1)
        self.assertEqual(len(result.a), 1)
        self.assertEqual(len(result.b), 1)
        self.assertIsInstance(result.a[0][0], Polynomial)
        C_init = result.C
        d_init = result.d / 2

        success, certificate = \
            self.cspace_free_polytope.FindSeparationCertificateGivenPolytope(
                C=C_init,
                d=d_init,
                ignored_collision_pairs=set(),
                options=bilinear_alternation_options.find_lagrangian_options)
        self.assertTrue(success)

        result = self.cspace_free_polytope.SearchWithBilinearAlternation(
            ignored_collision_pairs=set(),
            C_init=C_init,
            d_init=d_init,
            options=bilinear_alternation_options)
        self.assertGreaterEqual(len(result), 2)
        self.assertGreaterEqual(result[-1].num_iter, 0)

        success, certificates = \
            self.cspace_free_polytope.FindSeparationCertificateGivenPolytope(
                C=C_init,
                d=d_init,
                ignored_collision_pairs=set(),
                options=bilinear_alternation_options.find_lagrangian_options)
        self.assertTrue(success)
        geom1, geom2, certificate_result = certificates[0]
        self.assertGreaterEqual(certificate_result.plane_index, 0)
        self.assertGreaterEqual(
            len(certificate_result.positive_side_rational_lagrangians), 1)
        self.assertGreaterEqual(
            len(certificate_result.positive_side_rational_lagrangians), 1)
        self.assertEqual(len(certificate_result.a), 3)
        self.assertGreaterEqual(
            len(certificate_result.plane_decision_var_vals), 3)
        self.assertIsInstance(certificate_result.b, Polynomial)

        lagrangians = certificate_result.positive_side_rational_lagrangians[0]
        self.assertEqual(len(lagrangians.polytope), C_init.shape[0])
        self.assertEqual(len(lagrangians.s_lower), self.plant.num_positions())
        self.assertEqual(len(lagrangians.s_upper), self.plant.num_positions())
