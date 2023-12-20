import numpy as np
import scipy.sparse as sp
import unittest

import pydrake.planning as mut
from pydrake.common.test_utilities import numpy_compare
from pydrake.geometry.optimization import (HPolyhedron,
                                           Hyperrectangle,
                                           VPolytope,
                                           IrisOptions)
from pydrake.common import RandomGenerator

import textwrap
import scipy.sparse


class TestIrisFromCliqueCover(unittest.TestCase):
    def _make_robot_diagram(self):
        # Code taken from
        # bindings/pydrake/planning/test/collision_checker_test.py
        builder = mut.RobotDiagramBuilder()
        scene_yaml = textwrap.dedent("""
        directives:
        - add_model:
            name: box
            file: package://drake/multibody/models/box.urdf
        - add_model:
            name: ground
            file: package://drake/planning/test_utilities/collision_ground_plane.sdf  # noqa
        - add_weld:
            parent: world
            child: ground::ground_plane_box
        """)
        builder.parser().AddModelsFromString(scene_yaml, "dmd.yaml")
        model_instance_index = builder.plant().GetModelInstanceByName("box")
        robot_diagram = builder.Build()
        return (robot_diagram, model_instance_index)
    def _make_scene_graph_collision_checker(self, use_provider, use_function):
        # Code taken from
        # bindings/pydrake/planning/test/collision_checker_test.py
        self.assertFalse(use_provider and use_function)

        robot, index = self._make_robot_diagram()
        plant = robot.plant()
        checker_kwargs = dict(
            model=robot,
            robot_model_instances=[index],
            edge_step_size=0.125)

        if use_provider:
            checker_kwargs["distance_and_interpolation_provider"] = \
                mut.LinearDistanceAndInterpolationProvider(plant)
        if use_function:
            checker_kwargs["configuration_distance_function"] = \
                self._configuration_distance

        return mut.SceneGraphCollisionChecker(**checker_kwargs)

    def test_point_sampler_base_subclassable(self):
        class UniformHPolyhedronSamplerManual(mut.PointSamplerBase):
            def __init__(self, set, generator):
                mut.PointSamplerBase.__init__(self)
                self.set = set
                self.generator = generator

            def DoSamplePoints(self, num_points):
                ret = np.zeros((self.set.ambient_dimension(), num_points))
                for i in range(num_points):
                    ret[:, i] = self.set.UniformSample(self.generator)
                return ret

        A = np.array([[-1, 0], [0, -1], [1, 1]])
        b = np.array([0, 0, 1])
        set = HPolyhedron(A, b)
        generator = RandomGenerator(0)
        sampler = UniformHPolyhedronSamplerManual(set, generator)
        numpy_compare.assert_equal(sampler.set.A(), A)
        numpy_compare.assert_equal(sampler.set.b(), b)

        num_points = 3
        self.assertEqual(sampler.SamplePoints(num_points).shape, (2, 3))

    def test_uniform_set_sampler(self):
        A = np.array([[-1, 0], [0, -1], [1, 1]])
        b = np.array([0, 0, 1])
        set = HPolyhedron(A, b)
        generator = RandomGenerator(0)
        mut.UniformHPolyhedronSampler(set=set)
        sampler = mut.UniformHPolyhedronSampler(set=set, generator=generator)
        numpy_compare.assert_equal(sampler.Set().A(), A)
        numpy_compare.assert_equal(sampler.Set().b(), b)

        num_points = 3
        self.assertEqual(sampler.SamplePoints(num_points).shape, (2, 3))

        lb = np.array([0, 0, 0])
        ub = np.array([1, 1, 1])
        set = Hyperrectangle(lb, ub)
        mut.UniformHyperrectangleSampler(set=set)
        sampler = mut.UniformHyperrectangleSampler(set=set,
                                                   generator=generator)
        numpy_compare.assert_equal(sampler.Set().lb(), lb)
        numpy_compare.assert_equal(sampler.Set().ub(), ub)
        self.assertEqual(sampler.SamplePoints(num_points).shape, (3, 3))

    def test_rejection_sampler(self):
        lb = np.array([0, 0, 0])
        ub = np.array([1, 1, 1])
        set = Hyperrectangle(lb, ub)
        generator = RandomGenerator(0)
        base_sampler = mut.UniformHyperrectangleSampler(set=set,
                                                        generator=generator)
        sampler = mut.RejectionSampler(sampler=base_sampler,
                                       rejection_fun=lambda x: np.any(x > 0.5))
        num_points = 7
        self.assertEqual(sampler.SamplePoints(num_points=num_points).shape,
                         (3, num_points))

    def test_coverage_checker_base_subclassable(self):
        class BadCoverageChecker(mut.CoverageCheckerBase):
            def __init__(self):
                mut.CoverageCheckerBase.__init__(self)

            def DoCheckCoverage(self, current_sets):
                return any([c.IsEmpty() for c in current_sets])


        checker = BadCoverageChecker()
        sets = [Hyperrectangle(np.array([0, 0]), np.array([1, 1])),
                HPolyhedron(np.array([[-1, 0], [0, -1], [1, 1]]),
                            np.array([0, 0, 1]))]
        self.assertFalse(checker.CheckCoverage(sets=sets))
        # A trivially empty set. This test is here to ensure that
        # DoCheckCoverage is actually being accessed.
        sets.append(HPolyhedron(np.array([[1],[-1]]), np.array([[-1],[-1]])))
        self.assertTrue(checker.CheckCoverage(sets=sets))

    def test_bernoulli_coverage_checker(self):
        domain = Hyperrectangle(np.array([0, 0]), np.array([1, 1]))
        alpha = 0.5
        num_points_per_check = 10
        point_sampler = mut.UniformHyperrectangleSampler(
            set=domain, generator=RandomGenerator(0))
        num_threads = 3
        point_in_set_tol = 1e-10

        checker = mut.CoverageCheckerViaBernoulliTest(alpha=alpha,
                                      num_points_per_check=num_points_per_check,
                                      sampler=point_sampler,
                                      num_threads=num_threads,
                                      point_in_set_tol=point_in_set_tol)
        self.assertEqual(checker.get_alpha(), alpha)
        checker.set_alpha(alpha=0.25)
        self.assertEqual(checker.get_alpha(), 0.25)

        self.assertEqual(checker.get_num_points_per_check(),
                         num_points_per_check)
        checker.set_num_points_per_check(num_points_per_check=5)
        self.assertEqual(checker.get_num_points_per_check(), 5)

        self.assertEqual(checker.get_num_threads(), num_threads)
        checker.set_num_threads(num_threads=2)
        self.assertEqual(checker.get_num_threads(), 2)

        self.assertEqual(checker.get_point_in_set_tol(), point_in_set_tol)
        checker.set_point_in_set_tol(point_in_set_tol=1e-8)
        self.assertEqual(checker.get_point_in_set_tol(), 1e-8)

        # We have 100% coverage since the sets is the domain. For numerical
        # reasons we just ensure the sampled coverage is over 90%.
        sets = [domain]
        self.assertGreaterEqual(checker.GetSampledCoverageFraction(sets=sets), 0.9)
        self.assertTrue(checker.CheckCoverage(sets=sets))


        default_checker = mut.CoverageCheckerViaBernoulliTest(
            alpha=alpha, num_points_per_check=num_points_per_check,
            sampler=point_sampler
        )
        self.assertEqual(default_checker.get_num_threads(), -1)
        self.assertEqual(default_checker.get_point_in_set_tol(), 1e-8)

    def test_adjacency_matrix_base_is_subclassable(self):
        class FullAdjacencyMatrix(mut.AdjacencyMatrixBuilderBase):
            def __init__(self):
                mut.AdjacencyMatrixBuilderBase.__init__(self)
            def DoBuildAdjacencyMatrix(self, points):
                return np.ones((points.shape[1], points.shape[1]))

        builder = FullAdjacencyMatrix()
        x = np.linspace(0, 1, 10)
        points = np.vstack([x]*4)
        adjacency = builder.BuildAdjacencyMatrix(points)
        np.testing.assert_array_equal(adjacency.toarray(),
                                      np.ones((points.shape[1], points.shape[1])))
        self.assertIsInstance(adjacency, scipy.sparse.csc_matrix)

    def test_visibility_graph_builder(self):

        checker = self._make_scene_graph_collision_checker(True, False)
        builder = mut.VisibilityGraphBuilder(checker=checker, parallelize=True)
        plant = checker.model().plant()
        num_points = 2
        points = np.empty((plant.num_positions(), num_points))
        points[:, 0] = plant.GetPositions(checker.plant_context())
        points[:, 1] = points[:, 0]
        points[-1, 1] += 0.1
        A = builder.BuildAdjacencyMatrix(points=points)

        self.assertEqual(A.shape, (num_points, num_points))
        self.assertIsInstance(A, scipy.sparse.csc_matrix)

    def test_convex_set_from_clique_builder_base_subclassable(self):
        class VPolytopeBuilder(mut.ConvexSetFromCliqueBuilderBase):
            def __init__(self):
                mut.ConvexSetFromCliqueBuilderBase.__init__(self)

            def DoBuildConvexSet(self, clique_points):
                return VPolytope(clique_points)

        builder = VPolytopeBuilder()
        points = np.array([
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        vpolytope = builder.BuildConvexSet(points)
        self.assertIsInstance(vpolytope, VPolytope)
        np.testing.assert_array_equal(vpolytope.vertices(), points)

    def test_iris_region_from_clique_builder(self):
        options = IrisOptions()
        options.iteration_limit = 3
        obstacles = [HPolyhedron.MakeBox(np.zeros(2), np.ones(2))]
        domain = HPolyhedron.MakeBox(np.zeros(2), 2 * np.ones(2))
        rank_tol_for_lowner_john_ellipse = 1e-4

        builder = mut.IrisRegionFromCliqueBuilder(
            obstacles=obstacles,
            domain=domain,
            options=options,
            rank_tol_for_lowner_john_ellipse=rank_tol_for_lowner_john_ellipse
        )

        self.assertEqual(builder.get_options().iteration_limit, 3)
        self.assertEqual(len(builder.get_obstacles()), 1)
        np.testing.assert_array_equal(builder.get_domain().A(), domain.A())
        self.assertEqual(builder.get_rank_tol_for_lowner_john_ellipse(),
                         rank_tol_for_lowner_john_ellipse)

        options.iteration_limit = 1
        builder.set_options(options=options)
        self.assertEqual(builder.get_options().iteration_limit, 1)

        obstacles.append(HPolyhedron.MakeBox(np.ones(2), 2*np.ones(2)))
        builder.set_obstacles(obstacles)
        self.assertEqual(len(builder.get_obstacles()), 2)

        domain = HPolyhedron.MakeBox(np.zeros(2), 3 * np.ones(2))
        builder.set_domain(domain)
        np.testing.assert_array_equal(builder.get_domain().A(), domain.A())

        builder.set_rank_tol_for_lowner_john_ellipse(1e-6)
        self.assertEqual(builder.get_rank_tol_for_lowner_john_ellipse(), 1e-6)

        clique = np.array([
            [0.1, 0.4, 0.9],
            [1.1, 1.4, 1.3]
            ]
        )
        region = builder.BuildConvexSet(clique)
        self.assertIsInstance(region, HPolyhedron)
        self.assertTrue(
            region.ContainedIn(
                HPolyhedron.MakeBox(np.array([0,1]), np.array([1,3]))
            )
        )

