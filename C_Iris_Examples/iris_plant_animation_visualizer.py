import numpy as np
from pydrake.all import (HPolyhedron, AngleAxis,
                         VPolytope, Sphere, Ellipsoid, InverseKinematics,
                         RationalForwardKinematics, GeometrySet, Role,
                         RigidTransform, RotationMatrix,
                         Hyperellipsoid, Simulator, Box)
import mcubes

import C_Iris_Examples.visualization_utils as viz_utils

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
            cspace_free_polytope,
            cspace_frame,
            **kwargs):
        if plant.num_positions() > 3:
            raise ValueError(
                "Can't visualize the TC-Space of plants with more than 3-DOF")
        self.meshcat = StartMeshcat()
        self.meshcat.Delete()
        self.visualizer = MeshcatVisualizer.AddToBuilder(
            builder, scene_graph, self.meshcat)

        self.plant = plant
        self.builder = builder
        self.scene_graph = scene_graph
        self.viz_role = kwargs.get('viz_role', Role.kIllustration)

        self.diagram = self.builder.Build()
        self.diagram_context = self.diagram.CreateDefaultContext()


        self.plant_context = plant.GetMyMutableContextFromRoot(
            self.diagram_context)
        self.diagram.ForcedPublish(self.diagram_context)
        self.simulator = Simulator(
            self.diagram,
            self.diagram_context)
        self.simulator.Initialize()

        self.cspace_free_polytope = cspace_free_polytope

        # SceneGraph inspectors for highlighting geometry pairs.
        self.model_inspector = self.scene_graph.model_inspector()
        self.query = self.scene_graph.get_query_output_port().Eval(
            self.scene_graph.GetMyContextFromRoot(self.diagram_context))

        # Construct Rational Forward Kinematics for easy conversions.
        self.forward_kin = RationalForwardKinematics(plant)
        self.s_variables = sym.Variables(self.forward_kin.s())
        self.s_array = self.forward_kin.s()
        self.num_joints = self.plant.num_positions()

        # the point around which we construct the stereographic projection
        self.q_star = kwargs.get('q_star', np.zeros(self.num_joints))

        self.q_lower_limits = plant.GetPositionLowerLimits()
        self.s_lower_limits = self.forward_kin.ComputeSValue(
            self.q_lower_limits, self.q_star)
        self.q_upper_limits = plant.GetPositionUpperLimits()
        self.s_upper_limits = self.forward_kin.ComputeSValue(
            self.q_upper_limits, self.q_star)

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

        # The plane numbers which we wish to visualize.
        self._plane_indices_of_interest = []
        self.plane_indices = np.arange(
            0, len(cspace_free_polytope.separating_planes()))

        self.cspace_frame = cspace_frame

    def clear_plane_indices_of_interest(self):
        self._plane_indices_of_interest = []
        cur_q = self.plant.GetPositions(self.plant_context)
        self.show_res_q(cur_q)

    def add_plane_indices_of_interest(self, *elts):
        for e in elts:
            if e not in self._plane_indices_of_interest:
                self._plane_indices_of_interest.append(e)
        cur_q = self.plant.GetPositions(self.plant_context)
        self.show_res_q(cur_q)

    def remove_plane_indices_of_interest(self, *elts):
        self._plane_indices_of_interest[:] = (
            e for e in self._plane_indices_of_interest if e not in elts)
        cur_q = self.plant.GetPositions(self.plant_context)
        self.show_res_q(cur_q)

    def show_res_q(self, q, frame = None):
        self.plant.SetPositions(self.plant_context, q)
        in_collision = self.check_collision_q_by_ik(q)
        s = self.forward_kin.ComputeSValue(np.array(q), self.q_star)

        color = Rgba(1, 0.72, 0, 1) if in_collision else Rgba(0.24, 1, 0, 1)
        self.diagram.ForcedPublish(self.diagram_context)

        self.plot_cspace_points(s, name='/s', color=color, radius=0.05)
        if frame is not None:
            self.plot_cspace_points(s, name=f"/frame_{frame}/" +'/s', color=color, radius=0.05)

            self.meshcat.SetProperty(f"/frame_{frame}", 'visible', False)
        self.update_certificates(s, frame = frame)

    def show_res_s(self, s, frame = None):
        q = self.forward_kin.ComputeQValue(np.array(s), self.q_star)
        self.show_res_q(q, frame)

    def check_collision_q_by_ik(self, q, min_dist=1e-5):
        if np.all(q >= self.q_lower_limits) and \
                np.all(q <= self.q_upper_limits):
            return 1 - 1 * \
                float(self.collision_constraint.evaluator().CheckSatisfied(q, min_dist))
        else:
            return 1

    def check_collision_s_by_ik(self, s, min_dist=1e-5):
        s = np.array(s)
        q = self.forward_kin.ComputeQValue(s, self.q_star)
        return self.check_collision_q_by_ik(q, min_dist)

    def visualize_collision_constraint(self, **kwargs):
        if self.plant.num_positions() == 3:
            self._visualize_collision_constraint3d(**kwargs)
        else:
            self._visualize_collision_constraint2d(**kwargs)
        self.meshcat.SetTransform("/collision_constraint", self.cspace_frame)

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
                factor * self.s_lower_limits), tuple(
                factor * self.s_upper_limits), N, N, N, self.check_collision_s_by_ik, iso_surface)
        tri_drake = [SurfaceTriangle(*t) for t in triangles]
        self.meshcat.SetObject("/collision_constraint",
                                      TriangleSurfaceMesh(tri_drake, vertices),
                                      Rgba(1, 0, 0, 1), wireframe=wireframe)



    def _visualize_collision_constraint2d(self, factor=2, num_points=20):
        s0 = np.linspace(
            factor *
            self.s_lower_limits[0],
            factor *
            self.s_upper_limits[0],
            num_points)
        s1 = np.linspace(
            factor *
            self.s_lower_limits[0],
            factor *
            self.s_upper_limits[0],
            num_points)
        X, Y = np.meshgrid(s0, s1)
        Z = np.zeros_like(X)
        for i in range(num_points):
            for j in range(num_points):
                Z[i, j] = self.check_collision_s_by_ik(
                    np.array([X[i, j], Y[i, j]]))
                if Z[i, j] == 0:
                    Z[i, j] = np.nan
        Z = Z - 1
        viz_utils.plot_surface(
            self.meshcat,
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

    def plot_cspace_points(self, points, name, frame = None, **kwargs):
        points_trans = np.array([(self.cspace_frame @ RigidTransform(viz_utils.stretch_array_to_3d(p))).translation() for p in np.atleast_2d(points)])
        if len(points_trans.shape) == 1:
            viz_utils.plot_point(points_trans, self.meshcat, name, **kwargs)
            if frame is not None:
                viz_utils.plot_point(points_trans, self.meshcat, f"/frame_{frame}/" + name, **kwargs)
        else:
            for i, s in enumerate(points_trans):
                viz_utils.plot_point(
                    s, self.meshcat, name + f"/{i}", **kwargs)
                if frame is not None:
                    viz_utils.plot_point(points_trans, self.meshcat,
                                         f"/frame_{frame}/" + name + f"/{i}",
                                         **kwargs)


    def add_group_of_regions_and_certs_to_visualization(
            self, region_cert_color_tuples, group_name, **kwargs):
        # **kwargs are the ones for viz_utils.plot_polytopes
        # each element of region_and_certs_list is an (HPolyhedron,
        # SearchResult)
        self.region_certificate_groups[group_name] = region_cert_color_tuples
        self.update_region_visualization_by_group_name(group_name, **kwargs)


    def add_group_of_regions_to_visualization(
            self, region_color_tuples, group_name, **kwargs):
        # **kwargs are the ones for viz_utils.plot_polytopes
        self.region_certificate_groups[group_name] = [
            (region, None, color) for (
                region, color) in region_color_tuples]
        self.update_region_visualization_by_group_name(group_name, **kwargs)

    def update_region_visualization_by_group_name(self, name, frame = None, **kwargs):
        region_and_certificates_list = self.region_certificate_groups[name]
        for i, (r, _, color) in enumerate(region_and_certificates_list):
            viz_utils.plot_polytope(r, self.meshcat, f"/{name}/{i}",
                                    resolution=kwargs.get("resolution", 30),
                                    color=color,
                                    wireframe=kwargs.get("wireframe", True),
                                    random_color_opacity=kwargs.get("random_color_opacity", 0.7),
                                    fill=kwargs.get("fill", True),
                                    line_width=kwargs.get("line_width", 10),
                                    transformation=self.cspace_frame)

            name_prefix = f"/{name}/region_{i}"
            for plane_index in self._plane_indices_of_interest:#self.plane_indices:
                name = name_prefix + f"/plane_{plane_index}"
                self.meshcat.SetObject(name + "/plane",
                                                  Box(5, 5, 0.02),
                                                  Rgba(color.r(), color.g(), color.b(), 0.5))
                self.meshcat.SetProperty(name + "/plane", "visible", True)
                if frame is not None:
                    self.meshcat.SetObject(f"frame_{frame}" + name + "/plane",
                                           Box(5, 5, 0.02),
                                           Rgba(color.r(), color.g(), color.b(), 0.5))
                    self.meshcat.SetProperty(f"frame_{frame}" + name + "/plane", "visible", True)


    def animate_traj_s(self, traj, steps, runtime, sleep_time=0.1, time_step = 0.01):
        # loop
        idx = 0
        going_fwd = True
        time_points = np.linspace(0, traj.end_time(), steps)
        frame_count = 0
        # self.task_space_animiation = self.visualizer_task_space.get_mutable_recording()
        # self.cspace_animiation = self.visualizer_cspace.get_mutable_recording()
        for _ in range(runtime):
            # print(idx)
            t0 = time.time()
            s = traj.value(time_points[idx])
            self.show_res_s(s, frame=frame_count)
            self.diagram_context.SetTime(frame_count * time_step)
            self.diagram.ForcedPublish(self.diagram_context)
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
            # if pause > 0:
            #     time.sleep(pause)
        return frame_count


    def update_certificates(self, s, frame = None):
        for group_name, region_and_cert_list in self.region_certificate_groups.items():
            for i, (region, search_result, color) in enumerate(
                    region_and_cert_list):
                plane_color = Rgba(color.r(), color.g(), color.b(), 1) if color is not None else None
                name_prefix = f"/{group_name}/region_{i}"
                if region.PointInSet(s) and search_result is not None:
                    self.meshcat.SetProperty(name_prefix,
                                                        "visible", True)
                    if frame is not None:
                        self.meshcat.SetProperty(f"/frame_{frame}" + name_prefix,
                                                 "visible", True)
                    for plane_index in self.plane_indices:
                        if plane_index in self._plane_indices_of_interest:
                            self.plot_plane_by_index_at_s(
                                s, plane_index, search_result, plane_color,
                                name_prefix=name_prefix, frame = frame)
                        else:
                            self.meshcat.SetProperty(name_prefix + f"/plane_{plane_index}",
                                                                "visible", False)
                            if frame is not None:
                                self.update_region_visualization_by_group_name(group_name, frame=frame)
                                self.meshcat.SetProperty(f"/frame_{frame}" + name_prefix + f"/plane_{plane_index}",
                                                         "visible", False)
                else:
                    self.meshcat.SetProperty(name_prefix,
                                                        "visible", False)
                    if frame is not None:
                        self.meshcat.SetProperty(f"/frame_{frame}" + name_prefix,
                                                 "visible", False)

    def highlight_geometry_id(self, geom_id, color, name=None):
        if name is None:
            name = f"/id_{geom_id}"
        shape = self.model_inspector.GetShape(geom_id)
        X_WG = self.get_geom_id_pose_in_world(geom_id)
        self.meshcat.SetObject(name, shape, color)
        self.meshcat.SetTransform(name, X_WG)

    def get_geom_id_pose_in_world(self, geom_id):
        frame_id = self.model_inspector.GetFrameId(geom_id)
        X_FG = self.model_inspector.GetPoseInFrame(geom_id)
        X_WF = self.query.GetPoseInWorld(frame_id)
        return X_WF @ X_FG

    def plot_plane_by_index_at_s(
            self,
            s,
            plane_index,
            search_result,
            color,
            name_prefix="",
            frame = None):
        name = name_prefix + f"/plane_{plane_index}"
        sep_plane = self.cspace_free_polytope.separating_planes()[plane_index]

        geom1, geom2 = sep_plane.positive_side_geometry.id(),\
            sep_plane.negative_side_geometry.id()

        # highlight the geometry
        self.highlight_geometry_id(geom1, color, name + f"/{geom1}")
        self.highlight_geometry_id(geom2, color, name + f"/{geom2}")
        if frame is not None:
            self.highlight_geometry_id(geom1, color, name + f"/frame_{frame}/{geom1}")
            self.highlight_geometry_id(geom2, color, name + f"/frame_{frame}/{geom2}")

        env = {var_s: val_s for var_s, val_s in zip(
            self.cspace_free_polytope.rational_forward_kin().s(), s)}

        a = np.array([a_poly.Evaluate(env)
                     for a_poly in search_result.a[plane_index]])
        b = search_result.b[plane_index].Evaluate(env)

        expressed_body = self.plant.get_body(sep_plane.expressed_body)
        X_WE = self.plant.EvalBodyPoseInWorld(
            self.plant_context, expressed_body)
        X_EW = X_WE.inverse()
        X_WG1 = self.get_geom_id_pose_in_world(geom1)
        X_WG2 = self.get_geom_id_pose_in_world(geom2)
        p1 = (X_EW @ X_WG1).translation()
        p2 = (X_EW @ X_WG2).translation()

        mu = -b / (a.T @ (p2 - p1))
        offset = mu * (p2 - p1)
        axis = (a / np.linalg.norm(a))[:, np.newaxis]
        P = null_space(axis.T)
        R = np.hstack([P, axis])
        R = RotationMatrix(R)
        X_E_plane = RigidTransform(R, offset)

        self.meshcat.SetProperty(name + "/plane", "visible", True)
        self.meshcat.SetTransform(name + "/plane", X_WE @ X_E_plane)
        if frame is not None:
            self.meshcat.SetProperty(f"frame_{frame}" + name + "/plane", "visible", True)
            self.meshcat.SetTransform(f"frame_{frame}" + name + "/plane", X_WE @ X_E_plane)
