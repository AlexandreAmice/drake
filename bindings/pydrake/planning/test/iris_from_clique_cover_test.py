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
from pydrake.planning import MaxCliqueSolverViaMip
from pydrake.planning import (RobotDiagramBuilder,
                              SceneGraphCollisionChecker,
                              CollisionCheckerParams)

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
        sets.append(HPolyhedron(np.array([[1], [-1]]), np.array([[-1], [-1]])))
        self.assertTrue(checker.CheckCoverage(sets=sets))

    def test_bernoulli_coverage_checker(self):
        domain = Hyperrectangle(np.array([0, 0]), np.array([1, 1]))
        alpha = 0.5
        num_points_per_check = 10
        point_sampler = mut.UniformHyperrectangleSampler(
            set=domain, generator=RandomGenerator(0))
        num_threads = 3
        point_in_set_tol = 1e-10

        checker = mut.CoverageCheckerViaBernoulliTest(
            alpha=alpha,
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
        self.assertGreaterEqual(checker.GetSampledCoverageFraction(sets=sets),
                                0.9)
        self.assertTrue(checker.CheckCoverage(sets=sets))

        default_checker = mut.CoverageCheckerViaBernoulliTest(
            alpha=alpha,
            num_points_per_check=num_points_per_check,
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
                                      np.ones((points.shape[1],
                                               points.shape[1])))
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
                HPolyhedron.MakeBox(np.array([0, 1]), np.array([1, 3]))
            )
        )

    def test_approximate_convex_cover_from_clique_cover_options(self):
        options = mut.ApproximateConvexCoverFromCliqueCoverOptions()

        options.num_sampled_points = 10
        self.assertEqual(options.num_sampled_points, 10)

        options.minimum_clique_size = 5
        self.assertEqual(options.minimum_clique_size, 5)

    def test_approximate_convex_cover_from_clique_cover(self):
        class BadCoverageChecker(mut.CoverageCheckerBase):
            def __init__(self):
                mut.CoverageCheckerBase.__init__(self)

            def DoCheckCoverage(self, current_sets):
                return len(current_sets) > 0

        coverage_checker = BadCoverageChecker()

        class BadPointSampler(mut.PointSamplerBase):
            def __init__(self):
                mut.PointSamplerBase.__init__(self)

            def DoSamplePoints(self, num_points):
                clique = np.array([
                    [0.1, 0.4, 0.9, 0.8],
                    [1.1, 1.4, 1.3, 1.2]
                ]
                )
                return clique

        point_sampler = BadPointSampler()

        class FullAdjacencyMatrixBuilder(mut.AdjacencyMatrixBuilderBase):
            def __init__(self):
                mut.AdjacencyMatrixBuilderBase.__init__(self)

            def DoBuildAdjacencyMatrix(self, points):
                return np.ones((points.shape[1], points.shape[1]))

        adjacency_matrix_builder = FullAdjacencyMatrixBuilder()

        class BadCliqueSolver(mut.MaxCliqueSolverBase):
            def __init__(self):
                mut.MaxCliqueSolverBase.__init__(self)

            def DoSolveMaxClique(self, adjacency_matrix):
                return np.ones(adjacency_matrix.shape[0])

        max_clique_solver = BadCliqueSolver()

        class VPolytopeSetBuilder(mut.ConvexSetFromCliqueBuilderBase):
            def __init__(self):
                mut.ConvexSetFromCliqueBuilderBase.__init__(self)

            def DoBuildConvexSet(self, clique_points):
                return VPolytope(clique_points)

        num_set_builders = 10
        set_builders = [VPolytopeSetBuilder() for _ in range(num_set_builders)]
        options = mut.ApproximateConvexCoverFromCliqueCoverOptions()

        sets = mut.ApproximateConvexCoverFromCliqueCover(
            coverage_checker=coverage_checker,
            point_sampler=point_sampler,
            adjacency_matrix_builder=adjacency_matrix_builder,
            max_clique_solver=max_clique_solver,
            set_builders=set_builders,
            options=options)
        self.assertGreaterEqual(len(sets), 1)
        self.assertIsInstance(sets[0], VPolytope)

    def test_iris_from_clique_cover_options(self):
        options = mut.IrisFromCliqueCoverOptions()
        custom_iris_options = IrisOptions()
        custom_iris_options.iteration_limit = 25

        options.iris_options = custom_iris_options
        self.assertEqual(options.iris_options.iteration_limit,
                         custom_iris_options.iteration_limit)

        options.coverage_termination_threshold = 0.9
        self.assertEqual(options.coverage_termination_threshold, 0.9)

        options.num_points_per_coverage_check = int(1e8)
        self.assertEqual(options.num_points_per_coverage_check, int(1e8))

        options.minimum_clique_size = 10
        self.assertEqual(options.minimum_clique_size, 10)

        options.num_points_per_visibility_round = int(1e9)
        self.assertEqual(options.num_points_per_visibility_round, int(1e9))

        options.point_sampler = mut.UniformHyperrectangleSampler(
            set=Hyperrectangle(np.array([0, 0]), np.array([1, 1])),
        )
        self.assertIsInstance(options.point_sampler,
                              mut.UniformHyperrectangleSampler)

        options.num_builders = 10
        self.assertEqual(options.num_builders, 10)

        options.generator = RandomGenerator(0)
        val = RandomGenerator(0)()
        self.assertEqual(options.generator(), val)

        class BadCliqueSolver(mut.MaxCliqueSolverBase):
            def __init__(self):
                mut.MaxCliqueSolverBase.__init__(self)

            def DoSolveMaxClique(self, adjacency_matrix):
                return np.ones(adjacency_matrix.shape[0])

        self.assertIsInstance(options.max_clique_solver,
                              mut.MaxCliqueSolverViaMip)
        options2 = mut.IrisFromCliqueCoverOptions(BadCliqueSolver())
        self.assertIsInstance(options2.max_clique_solver, BadCliqueSolver)
    def test_iris_from_clique_cover(self):
        """
        A 1x1 box domain with a 0.45x0.45 obstacle in each corner. The free
        space is a cross composed to the union of
        Box1 with vertices (0.45,0), (0.45, 1), (0.55, 0), (0.55, 1)
        and
        Box2 with vertices (0, 0.45), (0, 0.55), (1, 0.45), (1, 0.55)
        """
        domain = HPolyhedron.MakeBox(np.array([0, 0]), np.array([1, 1]))

        obstacles = [
            HPolyhedron.MakeBox(np.array([0, 0]), np.array([0.45, 0.45])),
            HPolyhedron.MakeBox(np.array([0.55, 0]), np.array([1, 0.45])),
            HPolyhedron.MakeBox(np.array([0, 0.55]), np.array([0.45, 1])),
            HPolyhedron.MakeBox(np.array([0.55, 0.55]), np.array([1, 1]))]

        options = mut.IrisFromCliqueCoverOptions()
        options.num_builders = 3
        options.num_points_per_coverage_check = 100
        options.num_points_per_visibility_round = 100

        sets = mut.IrisFromCliqueCover(
            obstacles=obstacles, domain=domain, options=options, sets=[]
        )
        self.assertEqual(len(sets), 2)

        # Check that we can pass some sets and they are used.
        first_set = sets[0]
        sets = [sets[0]]
        sets = mut.IrisFromCliqueCover(
            obstacles=obstacles, domain=domain, options=options, sets=sets
        )
        np.testing.assert_array_equal(sets[0].A(), first_set.A())
        np.testing.assert_array_equal(sets[0].b(), first_set.b())
        self.assertEqual(len(sets), 2)

    def test_iris_in_configuration_space_from_clique_cover(self):
        # limits_urdf = """
        # <robot name="limits">
        #   <link name="movable">
        #     <collision>
        #       <geometry><box size="1 1 1"/></geometry>
        #     </collision>
        #   </link>
        #   <joint name="movable" type="prismatic">
        #     <axis xyz="1 0 0"/>
        #     <limit lower="-2" upper="2"/>
        #     <parent link="world"/>
        #     <child link="movable"/>
        #   </joint>
        # </robot>"""
        limits_urdf = """
        <robot name="boxes">
  <link name="fixed">
    <collision name="top_left">
      <origin rpy="0 0 0" xyz="-1 1 0"/>
      <geometry><box size="1 1 1"/></geometry>
    </collision>
    <collision name="top_right">
      <origin rpy="0 0 0" xyz="1 1 0"/>
      <geometry><box size="1 1 1"/></geometry>
    </collision>
    <collision name="bottom_left">
      <origin rpy="0 0 0" xyz="-1 -1 0"/>
      <geometry><box size="1 1 1"/></geometry>
    </collision>
    <collision name="bottom_right">
      <origin rpy="0 0 0" xyz="1 -1 0"/>
      <geometry><box size="1 1 1"/></geometry>
    </collision>
  </link>
  <joint name="fixed_link_weld" type="fixed">
    <parent link="world"/>
    <child link="fixed"/>
  </joint>
  <link name="movable">
    <collision name="sphere">
      <geometry><sphere radius="0.1"/></geometry>
    </collision>
  </link>
  <link name="for_joint"/>
  <joint name="x" type="prismatic">
    <axis xyz="1 0 0"/>
    <limit lower="-2" upper="2"/>
    <parent link="world"/>
    <child link="for_joint"/>
  </joint>
  <joint name="y" type="prismatic">
    <axis xyz="0 1 0"/>
    <limit lower="-2" upper="2"/>
    <parent link="for_joint"/>
    <child link="movable"/>
  </joint>
</robot>
"""
        params = dict(edge_step_size=0.125)
        builder = RobotDiagramBuilder()
        params["robot_model_instances"] = builder.parser().AddModelsFromString(
            limits_urdf, "urdf"
        )
        params["model"] = builder.Build()
        checker = SceneGraphCollisionChecker(**params)

        options = mut.IrisFromCliqueCoverOptions()
        options.num_builders = 1
        options.num_points_per_coverage_check = 10
        options.num_points_per_visibility_round = 10
        options.coverage_termination_threshold = 0.999
        options.iris_options

        # import logging
        # logger = logging.getLogger("drake")
        # logger.setLevel(logging.DEBUG)
        # from pydrake.common import configure_logging
        # configure_logging()

        sets = mut.IrisInConfigurationSpaceFromCliqueCover(
            checker=checker, options=options, sets=[]
        )

        self.assertGreaterEqual(len(sets), 1)
