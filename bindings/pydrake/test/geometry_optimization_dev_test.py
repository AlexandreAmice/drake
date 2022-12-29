import pydrake.geometry.optimization_dev as mut

import unittest

import numpy as np

from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.systems.framework import DiagramBuilder
from pydrake.solvers import MosekSolver, CommonSolverOption, SolverOptions
from pydrake.solvers import MathematicalProgram as mp



class TestGeometeryOptimizationDev(unittest.TestCase):
    def test_CspaceFreePolytope(self):
        limits_urdf = """
        <robot name="limits">
          <link name="movable">
            <collision>
              <geometry><box size="1 1 1"/></geometry>
            </collision>
          </link>
          <link name="unmovable">
            <collision>
              <geometry><box size="0.1 0.1 0.1"/></geometry>
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
                <pose> 2 0 0 0 0 0 </pose>
          </joint>
        </robot>"""
        builder = DiagramBuilder()
        plant, scene_graph = AddMultibodyPlantSceneGraph(builder, 0.0)
        Parser(plant).AddModelsFromString(limits_urdf, "urdf")
        plant.Finalize()
        diagram = builder.Build()

        cspace_free_polytope = mut.CspaceFreePolytope(plant, scene_graph, mut.SeparatingPlaneOrder.kAffine, np.zeros(1))

        solver_options = SolverOptions()
        solver_options.SetOption(CommonSolverOption.kPrintToConsole, 1)

        find_polytope_given_lagrangian_option = mut.FindPolytopeGivenLagrangianOptions()
        find_polytope_given_lagrangian_option.backoff_scale = 1e-3
        find_polytope_given_lagrangian_option.ellipsoid_margin_epsilon = 1e-6
        find_polytope_given_lagrangian_option.solver_id = MosekSolver.id()
        find_polytope_given_lagrangian_option.solver_options = solver_options
        find_polytope_given_lagrangian_option.s_inner_pts = np.zeros(1)

        find_separation_certificate_given_polytope_options = mut.FindSeparationCertificateGivenPolytopeOptions()
        find_separation_certificate_given_polytope_options.num_threads = -1
        find_separation_certificate_given_polytope_options.verbose = True
        find_separation_certificate_given_polytope_options.solver_id = MosekSolver.id()
        find_separation_certificate_given_polytope_options.terminate_at_failure = False
        find_separation_certificate_given_polytope_options.backoff_scale = 1e-5
        find_separation_certificate_given_polytope_options.solver_options = solver_options

        bilinear_alternation_options = mut.BilinearAlternationOptions()
        bilinear_alternation_options.max_iter = 4
        bilinear_alternation_options.convergence_tol = 1e-2
        bilinear_alternation_options.find_polytope_options = find_polytope_given_lagrangian_option
        bilinear_alternation_options.find_lagrangian_options = find_separation_certificate_given_polytope_options

        C_init = np.array([[1],[-1]])
        lim = 1e-2
        d_init = lim*np.ones((C_init.shape[0],1))

        (success, certificate) = cspace_free_polytope.FindSeparationCertificateGivenPolytope(C = C_init,
                                                                                             d = d_init,
                                                                                             ignored_collision_pairs = set(),
                                                                                             search_separating_margin = True,
                                                                                             options = find_separation_certificate_given_polytope_options)
        # self.assertTrue(success, "Separation Certificate not found")

        result = cspace_free_polytope.SearchWithBilinearAlternation(ignored_collision_pairs = set(),
                                                                    C_init=C_init, d_init=d_init,
                                                                    search_margin=True,
                                                                    options=bilinear_alternation_options)
        # self.assertGreaterEqual(result.num_iter, 2)