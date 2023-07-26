import numpy as np
from pydrake.all import (HPolyhedron, AngleAxis,
                         VPolytope, Sphere, Ellipsoid, InverseKinematics,
                         RationalForwardKinematics, GeometrySet, Role,
                         RigidTransform, RotationMatrix,
                         Hyperellipsoid, Simulator, Box)
import mcubes

import visualization_utils as viz_utils

import pydrake.symbolic as sym
from pydrake.all import MeshcatVisualizer, StartMeshcat, DiagramBuilder, \
    AddMultibodyPlantSceneGraph, TriangleSurfaceMesh, Rgba, SurfaceTriangle, Sphere
from scipy.linalg import null_space
import time


class IrisPlantVisualizer:
    def __init__(
            self,
            plant,
            builder,
            scene_graph,
            **kwargs):
        if plant.num_positions() > 3:
            raise ValueError(
                "Can't visualize the TC-Space of plants with more than 3-DOF")
        self.meshcat_task_space = StartMeshcat()
        self.meshcat_task_space.Delete()
        self.visualizer_task_space = MeshcatVisualizer.AddToBuilder(
            builder, scene_graph, self.meshcat_task_space)

        self.meshcat_cspace = StartMeshcat()
        self.meshcat_cspace.Delete()
        builder_cspace = DiagramBuilder()
        plant_cspace, scene_graph_cspace = AddMultibodyPlantSceneGraph(
            builder_cspace, time_step=0.0)
        plant_cspace.Finalize()

        self.visualizer_cspace = MeshcatVisualizer.AddToBuilder(
            builder_cspace, scene_graph_cspace, self.meshcat_cspace)


        self.plant = plant
        self.builder = builder
        self.scene_graph = scene_graph
        self.viz_role = kwargs.get('viz_role', Role.kIllustration)

        self.task_space_diagram = self.builder.Build()
        self.task_space_diagram_context = self.task_space_diagram.CreateDefaultContext()

        self.cspace_diagram = builder_cspace.Build()
        self.cspace_diagram_context = self.cspace_diagram.CreateDefaultContext()

        self.plant_context = plant.GetMyMutableContextFromRoot(
            self.task_space_diagram_context)
        self.task_space_diagram.ForcedPublish(self.task_space_diagram_context)
        self.simulator = Simulator(
            self.task_space_diagram,
            self.task_space_diagram_context)
        self.simulator.Initialize()


        # SceneGraph inspectors for highlighting geometry pairs.
        self.model_inspector = self.scene_graph.model_inspector()
        self.query = self.scene_graph.get_query_output_port().Eval(
            self.scene_graph.GetMyContextFromRoot(self.task_space_diagram_context))

        # Construct Rational Forward Kinematics for easy conversions.
        # self.forward_kin = RationalForwardKinematics(plant)
        # self.s_variables = sym.Variables(self.forward_kin.s())
        # self.s_array = self.forward_kin.s()
        self.num_joints = self.plant.num_positions()

        # the point around which we construct the stereographic projection
        self.q_star = kwargs.get('q_star', np.zeros(self.num_joints))

        self.q_lower_limits = plant.GetPositionLowerLimits()
        # self.s_lower_limits = self.forward_kin.ComputeSValue(
        #     self.q_lower_limits, self.q_star)
        self.q_upper_limits = plant.GetPositionUpperLimits()
        # self.s_upper_limits = self.forward_kin.ComputeSValue(
        #     self.q_upper_limits, self.q_star)

        # A dictionary mapping str -> (HPolyhedron, SearchResult, Color) where
        # SearchResult can be None. This is used for visualizing cspace regions
        # and their certificates in task space.
        self.region_certificate_groups = {}

        # Set up the IK object to enable visualization of the collision
        # constraint.
        self.ik = InverseKinematics(plant, self.plant_context)
        min_dist = 1e-5
        self.collision_constraint = self.ik.AddMinimumDistanceConstraint(
            min_dist, 1e-5)

    def show_res_q(self, q):
        self.plant.SetPositions(self.plant_context, q)
        in_collision = self.check_collision_q_by_ik(q)
        # s = self.forward_kin.ComputeSValue(np.array(q), self.q_star)

        color = Rgba(1, 0.72, 0, 1) if in_collision else Rgba(0.24, 1, 0, 1)
        self.task_space_diagram.ForcedPublish(self.task_space_diagram_context)

        self.plot_cspace_points(np.array(q), name='/q', color=color, radius=0.05)

    # def show_res_s(self, s):
    #     q = self.forward_kin.ComputeQValue(np.array(s), self.q_star)
    #     self.show_res_q(q)

    def check_collision_q_by_ik(self, q, min_dist=1e-5):
        if np.all(q >= self.q_lower_limits) and \
                np.all(q <= self.q_upper_limits):
            return 1 - 1 * \
                float(self.collision_constraint.evaluator().CheckSatisfied(q, min_dist))
        else:
            return 1

    # def check_collision_s_by_ik(self, s, min_dist=1e-5):
    #     s = np.array(s)
    #     q = self.forward_kin.ComputeQValue(s, self.q_star)
    #     return self.check_collision_q_by_ik(q, min_dist)

    def visualize_collision_constraint(self, **kwargs):
        if self.plant.num_positions() == 3:
            self._visualize_collision_constraint3d(**kwargs)
        else:
            self._visualize_collision_constraint2d(**kwargs)

    def _visualize_collision_constraint3d(
            self,
            N=50,
            factor=2,
            iso_surface=0.5,
            wireframe=True):
        """
        :param N: N is density of marchingcubes grid. Runtime scales cubically in N
        :return:
        """

        vertices, triangles = mcubes.marching_cubes_func(
            tuple(
                factor * self.q_lower_limits), tuple(
                factor * self.q_upper_limits), N, N, N, self.check_collision_q_by_ik, iso_surface)
        tri_drake = [SurfaceTriangle(*t) for t in triangles]
        self.meshcat_cspace.SetObject("/collision_constraint",
                                      TriangleSurfaceMesh(tri_drake, vertices),
                                      Rgba(1, 0, 0, 1), wireframe=wireframe)

    def _visualize_collision_constraint2d(self, factor=2, num_points=20):
        q0 = np.linspace(
            factor *
            self.q_lower_limits[0],
            factor *
            self.q_upper_limits[0],
            num_points)
        q1 = np.linspace(
            factor *
            self.q_lower_limits[0],
            factor *
            self.q_upper_limits[0],
            num_points)
        X, Y = np.meshgrid(q0, q1)
        Z = np.zeros_like(X)
        for i in range(num_points):
            for j in range(num_points):
                Z[i, j] = self.check_collision_q_by_ik(
                    np.array([X[i, j], Y[i, j]]))
                if Z[i, j] == 0:
                    Z[i, j] = np.nan
        Z = Z - 1
        viz_utils.plot_surface(
            self.meshcat_cspace,
            "/collision_constraint",
            X,
            Y,
            Z,
            Rgba(
                1,
                0,
                0,
                1))
        return Z

    def update_region_visualization_by_group_name(self, name, **kwargs):
        region_and_certificates_list = self.region_certificate_groups[name]
        for i, (r, _, color) in enumerate(region_and_certificates_list):
            viz_utils.plot_polytope(r, self.meshcat_cspace, f"/{name}/{i}",
                                    resolution=kwargs.get("resolution", 30),
                                    color=color,
                                    wireframe=kwargs.get("wireframe", True),
                                    random_color_opacity=kwargs.get("random_color_opacity", 0.7),
                                    fill=kwargs.get("fill", True),
                                    line_width=kwargs.get("line_width", 10))

    def update_region_visualization(self, **kwargs):
        for name in self.region_certificate_groups.keys():
            self.update_region_visualization_by_group_name(name, **kwargs)

    def add_group_of_regions_to_visualization(
            self, region_color_tuples, group_name, **kwargs):
        # **kwargs are the ones for viz_utils.plot_polytopes
        self.region_certificate_groups[group_name] = [
            (region, None, color) for (
                region, color) in region_color_tuples]
        self.update_region_visualization_by_group_name(group_name, **kwargs)

    def add_group_of_regions_and_certs_to_visualization(
            self, region_cert_color_tuples, group_name, **kwargs):
        self.update_region_visualization_by_group_name(group_name, **kwargs)

    def plot_cspace_points(self, points, name, **kwargs):
        if len(points.shape) == 1:
            viz_utils.plot_point(points, self.meshcat_cspace, name, **kwargs)
        else:
            for i, s in enumerate(points):
                viz_utils.plot_point(
                    s, self.meshcat_cspace, name + f"/{i}", **kwargs)

    def highlight_geometry_id(self, geom_id, color, name=None):
        if name is None:
            name = f"/id_{geom_id}"
        shape = self.model_inspector.GetShape(geom_id)
        X_WG = self.get_geom_id_pose_in_world(geom_id)
        self.meshcat_task_space.SetObject(name, shape, color)
        self.meshcat_task_space.SetTransform(name, X_WG)

    def get_geom_id_pose_in_world(self, geom_id):
        frame_id = self.model_inspector.GetFrameId(geom_id)
        X_FG = self.model_inspector.GetPoseInFrame(geom_id)
        X_WF = self.query.GetPoseInWorld(frame_id)
        return X_WF @ X_FG


    def animate_traj_s(self, traj, steps, runtime, idx_list = None, sleep_time = 0.1):
        # loop
        idx = 0
        going_fwd = True
        time_points = np.linspace(0, traj.end_time(), steps)
        frame_count = 0
        for _ in range(runtime):
            # print(idx)
            t0 = time.time()
            s = traj.value(time_points[idx])
            self.show_res_s(s)
            self.task_space_diagram_context.SetTime(frame_count * 0.01)
            self.task_space_diagram.ForcedPublish(self.task_space_diagram_context)
            self.cspace_diagram_context.SetTime(frame_count * 0.01)
            self.cspace_diagram.ForcedPublish(self.cspace_diagram_context)
            frame_count += 1
            if going_fwd:
                if idx + 1 < steps:
                    idx += 1
                else:
                    going_fwd = False
                    idx -= 1
            else:
                if idx - 1 >= 0:
                    idx -= 1
                else:
                    going_fwd = True
                    idx += 1
            t1 = time.time()
            pause = sleep_time - (t1 - t0)
            if pause > 0:
                time.sleep(pause)

    def save_meshcats(self, filename_prefix):
        with open(filename_prefix + "_cspace.html", "w") as f:
            f.write(self.meshcat_cspace.StaticHtml())
        with open(filename_prefix + "_task_space.html", "w") as f:
            f.write(self.meshcat_task_space.StaticHtml())
