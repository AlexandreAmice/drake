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
from pydrake.solvers import MathematicalProgram as mp
from pydrake.math import RigidTransform
from pydrake.solvers import MosekSolver, ScsSolver


class TestGeometeryOptimizationDev(unittest.TestCase):
    def setUp(self):
        unittest.TestCase.setUp(self)
        limits_urdf = """
                <robot name="limits">
                  <link name="movable">
                    <collision>
                      <geometry><box size="1 1 1"/></geometry>
                    </collision>
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
                        <origin xyz="20 0 0"/>
                  </joint>
                </robot>"""

        builder = DiagramBuilder()
        self.plant, self.scene_graph = AddMultibodyPlantSceneGraph(
            builder, MultibodyPlant(time_step=0.01))
        Parser(self.plant).AddModelsFromString(limits_urdf, "urdf")

        # self.body0 = self.plant.GetBodyByName("rail_base")
        # self.body1 = self.plant.GetBodyByName("pendulum")

        # proximity_properties = ProximityProperties()
        #
        #
        # # Register more collision geometries to the pendulum.
        # self.cylinder = self.plant.RegisterCollisionGeometry(
        #     self.body1, RigidTransform(np.zeros(3)),
        #     Cylinder(0.1, 0.5)
        #
        # )

        self.plant.Finalize()

        diagram = builder.Build()
        # test constructor
        self.cspace_free_polytope = mut.CspaceFreePolytope(
            plant=self.plant,
            scene_graph=self.scene_graph,
            plane_order=mut.SeparatingPlaneOrder.kAffine,
            q_star=np.zeros(1))

        movable = self.plant.GetBodyByName("movable")
        unmovable = self.plant.GetBodyByName("unmovable")
        # print(movable.body_frame().CalcPoseInWorld(
        #     self.plant.CreateDefaultContext()))
        # print(unmovable.body_frame().CalcPoseInWorld(
        #     self.plant.CreateDefaultContext()))

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
                self.assertTrue(geom.type() in geom_type_possible_values)
                self.assertTrue(type(geom.geometry())
                                in geom_shape_possible_values)
                self.assertTrue(
                    geom.body_index() in collision_geometries.keys())
                self.assertGreater(geom.num_rationals(), 0)
                self.assertIsInstance(geom.X_BG(), RigidTransform)
                self.assertIsInstance(geom.id(), GeometryId)

        # Check that the plane sides are properly enumerated.
        plane_side_possible_values = [mut.PlaneSide.kPositive,
                                      mut.PlaneSide.kNegative]

    def test_separating_plane(self):
        # Check that the plane orders are properly enumerated.
        possible_orders = [mut.SeparatingPlaneOrder.kAffine]

    def test_CspaceFreePolytope_Options(self):
        solver_options = SolverOptions()
        solver_options.SetOption(CommonSolverOption.kPrintToConsole, 1)

        find_polytope_given_lagrangian_option = mut.FindPolytopeGivenLagrangianOptions()
        self.assertIsNone(find_polytope_given_lagrangian_option.backoff_scale)
        self.assertEquals(
            find_polytope_given_lagrangian_option.ellipsoid_margin_epsilon, 1e-5)
        self.assertEquals(
            find_polytope_given_lagrangian_option.solver_id,
            MosekSolver.id())
        self.assertIsNone(find_polytope_given_lagrangian_option.solver_options)
        self.assertIsNone(find_polytope_given_lagrangian_option.s_inner_pts)
        self.assertTrue(
            find_polytope_given_lagrangian_option.search_s_bounds_lagrangians)
        # TODO (AlexandreAmice) uncomment this once the margin costs are properly enumerated.
        # self.assertEqual(find_polytope_given_lagrangian_option.ellipsoid_margin_cost)

        find_polytope_given_lagrangian_option.backoff_scale = 1e-3
        find_polytope_given_lagrangian_option.ellipsoid_margin_epsilon = 1e-6
        find_polytope_given_lagrangian_option.solver_id = ScsSolver.id()
        find_polytope_given_lagrangian_option.solver_options = solver_options
        find_polytope_given_lagrangian_option.s_inner_pts = np.zeros((2, 1))
        find_polytope_given_lagrangian_option.search_s_bounds_lagrangians = False
        # TODO (AlexandreAmice) uncomment this once the margin costs are properly enumerated.
        # find_polytope_given_lagrangian_option.ellipsoid_margin_cost = True
        self.assertEquals(
            find_polytope_given_lagrangian_option.backoff_scale, 1e-3)
        self.assertEquals(
            find_polytope_given_lagrangian_option.ellipsoid_margin_epsilon, 1e-6)
        self.assertEquals(
            find_polytope_given_lagrangian_option.solver_id,
            ScsSolver.id())
        self.assertEqual(
            find_polytope_given_lagrangian_option.solver_options.common_solver_options()[
                CommonSolverOption.kPrintToConsole], 1)
        np.testing.assert_array_almost_equal(
            find_polytope_given_lagrangian_option.s_inner_pts, np.zeros(
                (2, 1)), 1e-5)
        self.assertFalse(
            find_polytope_given_lagrangian_option.search_s_bounds_lagrangians)
        # TODO (AlexandreAmice) uncomment this once the margin costs are properly enumerated.
        # self.assertEqual(find_polytope_given_lagrangian_option.ellipsoid_margin_cost)

        find_separation_certificate_given_polytope_options = mut.FindSeparationCertificateGivenPolytopeOptions()
        self.assertEqual(
            find_separation_certificate_given_polytope_options.num_threads, -1)
        self.assertFalse(
            find_separation_certificate_given_polytope_options.verbose)
        self.assertEqual(
            find_separation_certificate_given_polytope_options.solver_id,
            MosekSolver.id())
        self.assertTrue(
            find_separation_certificate_given_polytope_options.terminate_at_failure)
        self.assertIsNone(
            find_separation_certificate_given_polytope_options.solver_options)
        self.assertFalse(
            find_separation_certificate_given_polytope_options.ignore_redundant_C)

        num_threads = 1
        find_separation_certificate_given_polytope_options.num_threads = num_threads
        find_separation_certificate_given_polytope_options.verbose = True
        find_separation_certificate_given_polytope_options.solver_id = ScsSolver.id()
        find_separation_certificate_given_polytope_options.terminate_at_failure = False
        find_separation_certificate_given_polytope_options.solver_options = solver_options
        find_separation_certificate_given_polytope_options.ignore_redundant_C = True
        self.assertEqual(
            find_separation_certificate_given_polytope_options.num_threads,
            num_threads)
        self.assertTrue(
            find_separation_certificate_given_polytope_options.verbose)
        self.assertEqual(
            find_separation_certificate_given_polytope_options.solver_id,
            ScsSolver.id())
        self.assertFalse(
            find_separation_certificate_given_polytope_options.terminate_at_failure)
        self.assertEqual(
            find_separation_certificate_given_polytope_options.solver_options.common_solver_options()[
                CommonSolverOption.kPrintToConsole], 1)
        self.assertTrue(
            find_separation_certificate_given_polytope_options.ignore_redundant_C)

        bilinear_alternation_options = mut.BilinearAlternationOptions()
        self.assertEqual(bilinear_alternation_options.max_iter, 10)
        self.assertAlmostEqual(
            bilinear_alternation_options.convergence_tol, 1e-3, 1e-10)
        self.assertAlmostEqual(
            bilinear_alternation_options.ellipsoid_scaling, 0.99, 1e-10)
        self.assertTrue(
            bilinear_alternation_options.find_polytope_options.search_s_bounds_lagrangians)
        self.assertFalse(
            bilinear_alternation_options.find_lagrangian_options.verbose)

        bilinear_alternation_options.max_iter = 4
        bilinear_alternation_options.convergence_tol = 1e-2
        bilinear_alternation_options.find_polytope_options = find_polytope_given_lagrangian_option
        bilinear_alternation_options.find_lagrangian_options = find_separation_certificate_given_polytope_options
        bilinear_alternation_options.ellipsoid_scaling = 0.5
        self.assertEqual(bilinear_alternation_options.max_iter, 4)
        self.assertAlmostEqual(
            bilinear_alternation_options.convergence_tol, 1e-2, 1e-10)
        self.assertAlmostEqual(
            bilinear_alternation_options.ellipsoid_scaling, 0.5, 1e-10)
        self.assertFalse(
            bilinear_alternation_options.find_polytope_options.search_s_bounds_lagrangians)
        self.assertTrue(
            bilinear_alternation_options.find_lagrangian_options.verbose)

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
        binary_search_options.find_lagrangian_options = find_separation_certificate_given_polytope_options
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

    def test_CspaceFreePolytope(self):

        C_init = np.array([[1], [-1]])
        lim = 1e-2
        d_init = lim * np.ones((C_init.shape[0], 1))

        bilinear_alternation_options = mut.BilinearAlternationOptions()
        binary_search_options = mut.BinarySearchOptions()
        bilinear_alternation_options.find_lagrangian_options.verbose = True
        binary_search_options.find_lagrangian_options.verbose = True

        success, certificate = self.cspace_free_polytope.FindSeparationCertificateGivenPolytope(
            C=C_init,
            d=d_init,
            ignored_collision_pairs=set(),
            options=bilinear_alternation_options.find_lagrangian_options)
        self.assertTrue(success)


        result = self.cspace_free_polytope.BinarySearch(
            ignored_collision_pairs=set(),
            C=C_init,
            d=d_init,
            s_center= np.zeros(1),
            options=binary_search_options
        )


        result = self.cspace_free_polytope.SearchWithBilinearAlternation(
            ignored_collision_pairs=set(),
            C_init=C_init,
            d_init=d_init,
            options=bilinear_alternation_options)
        for r in result:
            print(r.num_iter)
