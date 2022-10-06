import time

import numpy as np
import scipy
from pydrake.all import (HPolyhedron,
                         VPolytope, Sphere, Ellipsoid, InverseKinematics,
                         RationalForwardKinematics, GeometrySet, Role,
                         RigidTransform, RotationMatrix,
                         Hyperellipsoid, Simulator, Box)
from pydrake.all import GetVertices
from functools import partial
import mcubes
import C_Iris_Examples.visualizations_utils as viz_utils
import pydrake.symbolic as sym
from IPython.display import display
from scipy.spatial import Delaunay, ConvexHull
from scipy.linalg import block_diag

from pydrake.all import MeshcatVisualizerCpp, StartMeshcat, MeshcatVisualizer, DiagramBuilder, \
    AddMultibodyPlantSceneGraph, TriangleSurfaceMesh, Rgba, SurfaceTriangle, Sphere




class IrisPlantVisualizer:
    def __init__(self, plant, builder, scene_graph, **kwargs):
        self.meshcat1 = StartMeshcat()
        self.meshcat2 = StartMeshcat()
        self.meshcat1.Delete()
        self.meshcat2.Delete()
        self.vis = MeshcatVisualizerCpp.AddToBuilder(builder, scene_graph, self.meshcat1)


        builder2 = DiagramBuilder()
        plant2, scene_graph2 = AddMultibodyPlantSceneGraph(builder2, time_step=0.0)
        self.vis2 = MeshcatVisualizerCpp.AddToBuilder(builder2, scene_graph2, self.meshcat2)

        self.plant = plant

        self.builder = builder
        self.scene_graph = scene_graph

        # Construct Rational Forward Kinematics
        self.forward_kin = RationalForwardKinematics(plant)
        self.s_variables = sym.Variables(self.forward_kin.t())
        self.s_array = self.forward_kin.t()
        self.num_joints = self.plant.num_positions()
        # the point around which we construct the stereographic projection
        self.q_star = kwargs.get('q_star', np.zeros(self.num_joints))
        self.q_lower_limits = plant.GetPositionLowerLimits()
        self.s_lower_limits = self.forward_kin.ComputeTValue(self.q_lower_limits, self.q_star)
        tmp = -1
        self.q_lower_limits = viz_utils.stretch_array_to_3d(self.q_lower_limits,tmp)
        self.s_lower_limits = viz_utils.stretch_array_to_3d(self.s_lower_limits,tmp)

        self.q_upper_limits = plant.GetPositionUpperLimits()
        self.s_upper_limits = self.forward_kin.ComputeTValue(self.q_upper_limits, self.q_star)
        self.q_upper_limits = viz_utils.stretch_array_to_3d(self.q_upper_limits)
        self.s_upper_limits = viz_utils.stretch_array_to_3d(self.s_upper_limits)

        self.viz_role = kwargs.get('viz_role', Role.kIllustration)

        self.diagram = self.builder.Build()



        builder2.Build()
        # self.meshcat1.load()
        self.diagram_context = self.diagram.CreateDefaultContext()
        self.plant_context = plant.GetMyMutableContextFromRoot(self.diagram_context)
        self.diagram.Publish(self.diagram_context)
        self.simulator = Simulator(self.diagram, self.diagram_context)
        self.simulator.Initialize()

        self.ik = InverseKinematics(plant, self.plant_context)
        min_dist = 1e-5
        self.collision_constraint = self.ik.AddMinimumDistanceConstraint(min_dist, 0.01)
        self.col_func_handle = partial(self.eval_cons, c=self.collision_constraint, tol=min_dist)
        self.col_func_handle_rational = partial(self.eval_cons_rational)

        # construct collision pairs
        self.query = self.scene_graph.get_query_output_port().Eval(
            self.scene_graph.GetMyContextFromRoot(self.diagram_context))
        self.inspector = self.query.inspector()
        self.pairs = self.inspector.GetCollisionCandidates()

        # transforms for plotting planes
        # only gets kProximity pairs. Might be more efficient?
        # geom_ids = inspector.GetGeometryIds(GeometrySet(inspector.GetAllGeometryIds()), Role.kProximity)
        pair_set = set()
        for p in self.pairs:
            pair_set.add(p[0])
            pair_set.add(p[1])
        self.geom_ids = self.inspector.GetGeometryIds(GeometrySet(list(pair_set)))
        self.link_poses_by_body_index_rat_pose = self.forward_kin.CalcLinkPoses(self.q_star,
                                                                                self.plant.world_body().index())
        self.X_WA_list = [p.asRigidTransformExpr() for p in self.link_poses_by_body_index_rat_pose]
        self.body_indexes_by_geom_id = {geom:
                                            plant.GetBodyFromFrameId(self.inspector.GetFrameId(geom)).index() for geom
                                        in
                                        self.geom_ids}
        self.hpoly_sets_in_self_frame_by_geom_id = {
            geom: self.MakeFromHPolyhedronSceneGraph(self.query, geom, self.inspector.GetFrameId(geom))
            for geom in self.geom_ids}
        self.vpoly_sets_in_self_frame_by_geom_id = {
            geom: self.MakeFromVPolytopeSceneGraph(self.query, geom, self.inspector.GetFrameId(geom))
            for geom in self.geom_ids}

        self.s_space_vertex_world_position_by_geom_id = {}
        for geom in self.geom_ids:
            VPoly = self.vpoly_sets_in_self_frame_by_geom_id[geom]
            num_verts = VPoly.vertices().shape[1]
            X_WA = self.X_WA_list[int(self.body_indexes_by_geom_id[geom])]
            R_WA = X_WA.rotation().matrix()
            p_WA = X_WA.translation()
            vert_pos = R_WA @ (VPoly.vertices()) + np.repeat(p_WA[:, np.newaxis], num_verts, 1)
            self.s_space_vertex_world_position_by_geom_id[geom] = vert_pos

        #plotting planes setup
        x = np.linspace(-1, 1, 3)
        y = np.linspace(-1, 1, 3)
        width = 0.1
        z = np.linspace(-width, width, 3)
        verts = []

        for idxx in range(len(x)):
            for idxy in range(len(y)):
                verts.append(np.array([x[idxx], y[idxy]]))

        self.tri = Delaunay(verts)
        self.plane_triangles = self.tri.simplices
        tmp = self.tri.points[:, :]
        self.plane_verts = np.concatenate((tmp, 0 * tmp[:, 0].reshape(-1, 1)), axis=1)
        self.plane_verts = np.vstack([self.plane_verts,
                                      np.concatenate((tmp[::-1], 0.1 * np.ones_like(tmp[:, 0].reshape(-1, 1))), axis=1)])

        self.box = Box(2, 2, 0.02)

        #region -> (collision -> plane dictionary)
        self.region_to_collision_pair_to_plane_dictionary = None


        self.cube_tri = np.array([[0,2,1],[0,3,2],
                     [4,6,5],[4,7,6],
                     [4,3,7],[3,0,7],
                     [0,6,7],[0,1,6],
                     [1,5,6],[1,2,5],
                     [2,4,5],[2,3,4]])
        self.cube_tri_drake = [SurfaceTriangle(*t) for t in self.cube_tri]

        self._certified_region_solution_list = []
        self._collision_pairs_of_interest = []
        self._region_to_planes_of_interest_dict = {}

        self.color_dict = None

        self.do_viz_2 = plant.num_positions() <= 3

    @property
    def collision_pairs_of_interest(self):
        return self._collision_pairs_of_interest

    @collision_pairs_of_interest.setter
    def collision_pairs_of_interest(self, pair_list):
        self._collision_pairs_of_interest = pair_list
        self._refresh_planes_of_interest()

    @property
    def certified_region_solution_list(self):
        return self._certified_region_solution_list

    @certified_region_solution_list.setter
    def certified_region_solution_list(self, solution_list):
        self._certified_region_solution_list = solution_list
        self._refresh_planes_of_interest()

    def _refresh_planes_of_interest(self):
        # keeps a data structure mapping a region to a plane of interest so that we don't
        # need to search for the planes of interest every time.
        self._region_to_planes_of_interest_dict = {k: [] for k in self.certified_region_solution_list}
        for certified_region_solution in self.certified_region_solution_list:
            for i, plane in enumerate(certified_region_solution.separating_planes):
                A = plane.positive_side_geometry.id() if plane.positive_side_geometry is not None else None
                B = plane.negative_side_geometry.id() if plane.negative_side_geometry is not None else None
                for gid_pairs in self._collision_pairs_of_interest:
                    if (A, B) == gid_pairs or (B, A) == gid_pairs:
                        self._region_to_planes_of_interest_dict[certified_region_solution].append(plane)


    def jupyter_cell(self,):
        display(self.vis.jupyter_cell())
        if self.do_viz_2:
            display(self.vis2.jupyter_cell())

    def eval_cons(self, q, c, tol):
        if np.all(q >= self.q_lower_limits[:self.num_joints]) and \
                np.all(q <= self.q_upper_limits[:self.num_joints]):
            return 1 - 1 * float(c.evaluator().CheckSatisfied(q, tol))
        else:
            return 1

    def eval_cons_rational(self, *s):
        s = np.array(s[:self.num_joints])
        q = self.forward_kin.ComputeQValue(np.array(s), self.q_star)
        return self.col_func_handle(q)

    def visualize_collision_constraint(self, N = 50, factor = 2, iso_surface = 0.5, wireframe = True):
        """
        :param N: N is density of marchingcubes grid. Runtime scales cubically in N
        :return:
        """

        vertices, triangles = mcubes.marching_cubes_func(tuple(factor*self.s_lower_limits),
                                                         tuple(factor*self.s_upper_limits),
                                                         N, N, N, self.col_func_handle_rational, iso_surface)
        tri_drake = [SurfaceTriangle(*t) for t in triangles]
        self.meshcat2.SetObject("/collision_constraint",
                                      TriangleSurfaceMesh(tri_drake, vertices),
                                      Rgba(1, 0, 0, 1), wireframe=wireframe)

    def plot_path(self, verts, width = 0.01, sphere_rad = 0.01, color = Rgba(0.0, 0.0, 1, 1), prefix = ""):
        self.meshcat2.SetObject(f"/{prefix}path/start",
                                Sphere(sphere_rad),
                                color)
        self.meshcat2.SetTransform(f"/{prefix}path/start",
                                   RigidTransform(RotationMatrix(),
                                                  viz_utils.stretch_array_to_3d(verts[:, 0][:, np.newaxis])))
        self.meshcat2.SetObject(f"/{prefix}path/end",
                                Sphere(sphere_rad),
                                color)
        self.meshcat2.SetTransform(f"/{prefix}path/end",
                                   RigidTransform(RotationMatrix(),
                                                  viz_utils.stretch_array_to_3d(verts[:, -1][:, np.newaxis])))
        for idx in range(verts.shape[1] - 1):
            v1 = viz_utils.stretch_array_to_3d(verts[:, idx])
            v2 = viz_utils.stretch_array_to_3d(verts[:, idx + 1])
            name = f"/{prefix}path/rm/line{idx}"
            self.meshcat2.SetLine(name, np.hstack([v1[:, np.newaxis], v2[:, np.newaxis]]),
                                        line_width=width, rgba=color)

    def plot_surface(self, meshcat,
                     path,
                     X,
                     Y,
                     Z,
                     rgba=Rgba(.87, .6, .6, 1.0),
                     wireframe=False,
                     wireframe_line_width=1.0):
        # taken from https://github.com/RussTedrake/manipulation/blob/346038d7fb3b18d439a88be6ed731c6bf19b43de/manipulation/meshcat_cpp_utils.py#L415
        (rows, cols) = Z.shape
        assert (np.array_equal(X.shape, Y.shape))
        assert (np.array_equal(X.shape, Z.shape))

        vertices = np.empty((rows * cols, 3), dtype=np.float32)
        vertices[:, 0] = X.reshape((-1))
        vertices[:, 1] = Y.reshape((-1))
        vertices[:, 2] = Z.reshape((-1))

        # Vectorized faces code from https://stackoverflow.com/questions/44934631/making-grid-triangular-mesh-quickly-with-numpy  # noqa
        faces = np.empty((rows - 1, cols - 1, 2, 3), dtype=np.uint32)
        r = np.arange(rows * cols).reshape(rows, cols)
        faces[:, :, 0, 0] = r[:-1, :-1]
        faces[:, :, 1, 0] = r[:-1, 1:]
        faces[:, :, 0, 1] = r[:-1, 1:]
        faces[:, :, 1, 1] = r[1:, 1:]
        faces[:, :, :, 2] = r[1:, :-1, None]
        faces.shape = (-1, 3)

        # TODO(Russ): support per vertex / Colormap colors.
        meshcat.SetTriangleMesh(path, vertices.T, faces.T, rgba, wireframe,
                                wireframe_line_width)

    def visualize_collision_constraint2d(self, factor=2, num_points=20):
        s0 = np.linspace(factor * self.s_lower_limits[0], factor * self.s_upper_limits[0], num_points)
        s1 = np.linspace(factor * self.s_lower_limits[0], factor * self.s_upper_limits[0], num_points)
        X, Y = np.meshgrid(s0, s1)
        Z = np.zeros_like(X)
        for i in range(num_points):
            for j in range(num_points):
                Z[i, j] = self.eval_cons_rational(X[i, j], Y[i, j])
                if Z[i, j] == 0:
                    Z[i, j] = np.nan
        Z = Z-1
        self.plot_surface(self.meshcat2, "/collision_constraint", X, Y, Z, Rgba(1,0,0,1))
        return Z

    def plot_regions(self, regions, ellipses = None,
                     region_suffix = '', colors = None,
                     wireframe = True,
                     opacity = 0.7,
                     fill = True,
                     line_width = 10,
                     darken_factor = .2,
                     el_opacity = 0.3):
        if colors is None:
            colors = viz_utils.n_colors_random(len(regions), rgbs_ret=True)

        for i, region in enumerate(regions):
            c = Rgba(*[col/255 for col in colors[i]],opacity)
            prefix = f"/iris/regions{region_suffix}/{i}"
            name = prefix + "/hpoly"
            if region.ambient_dimension() == 3:
                self.plot_hpoly3d(self.meshcat2, name, region,
                                  c, wireframe = wireframe, resolution = 30)
            elif region.ambient_dimension() == 2:
                self.plot_hpoly2d(self.meshcat2, name,
                                  region, *[col/255 for col in colors[i]],
                                  a=opacity,
                             line_width=line_width,
                             fill=fill)

            if ellipses is not None:
                name = prefix + "/ellipse"
                c = Rgba(*[col/255*(1-darken_factor) for col in colors[i]],el_opacity)
                self.plot_ellipse(self.meshcat2, name,
                                  ellipses[i], c)


    def plot_seedpoints(self, seed_points):
        for i in range(seed_points.shape[0]):
            self.meshcat2.SetObject(f"/iris/seedpoints/seedpoint{i}",
                                   Sphere(0.05),
                                   Rgba(0.06, 0.0, 0, 1))
            s = np.zeros(3)
            s[:len(seed_points[i])] = seed_points[i]
            self.meshcat2.SetTransform(f"/iris/seedpoints/seedpoint{i}",
                                       RigidTransform(RotationMatrix(),
                                                      s))

    def get_plot_poly_mesh(self, region, resolution):

        def inpolycheck(q0, q1, q2, A, b):
            q = np.array([q0, q1, q2])
            res = np.min(1.0 * (A @ q - b <= 0))
            # print(res)
            return res

        aabb_max, aabb_min = viz_utils.get_AABB_limits(region)

        col_hand = partial(inpolycheck, A=region.A(), b=region.b())
        vertices, triangles = mcubes.marching_cubes_func(tuple(aabb_min),
                                                         tuple(aabb_max),
                                                         resolution,
                                                         resolution,
                                                         resolution,
                                                         col_hand,
                                                         0.5)
        tri_drake = [SurfaceTriangle(*t) for t in triangles]
        return vertices, tri_drake

    def plot_hpoly3d(self, meshcat, name, hpoly, color, wireframe = True, resolution = 30):
        verts, triangles = self.get_plot_poly_mesh(hpoly,
                                                   resolution=resolution)
        meshcat.SetObject(name, TriangleSurfaceMesh(triangles, verts),
                                color, wireframe=wireframe)

    def plot_hpoly2d(self, meshcat, name, hpoly, r = 0., g = 0., b = 1., a = 0.,
                     line_width = 8,
                     fill = False):
        # plot boundary
        vpoly = VPolytope(hpoly)
        verts = vpoly.vertices()
        hull = ConvexHull(verts.T)
        inds = np.append(hull.vertices, hull.vertices[0])
        hull_drake = verts.T[inds, :].T
        hull_drake3d = np.vstack([hull_drake, np.zeros(hull_drake.shape[1])])
        meshcat.SetLine(name, hull_drake3d,
                        line_width=line_width, rgba=Rgba(r, g, b, 1))
        if fill:
            width = 0.5
            C = block_diag(hpoly.A(), np.array([-1, 1])[:, np.newaxis])
            d = np.append(hpoly.b(), width * np.ones(2))
            hpoly_3d = HPolyhedron(C,d)
            self.plot_hpoly3d(meshcat, name+"/fill",
                              hpoly_3d, Rgba(r, g, b, a),
                              wireframe=False)

    def plot_ellipse(self,  meshcat, name, ellipse, color):
        if ellipse.A().shape[0] == 2:
            ellipse = Hyperellipsoid(block_diag(ellipse.A(), 1),
                                     np.append(ellipse.center(), 0))
        shape, pose = ellipse.ToShapeWithPose()

        meshcat.SetObject(name, shape, color)
        meshcat.SetTransform(name, pose)


    def showres(self,q, idx_list = None):
        self.plant.SetPositions(self.plant_context, q)
        col = self.col_func_handle(q)
        s = self.forward_kin.ComputeTValue(np.array(q), self.q_star)
        s = viz_utils.stretch_array_to_3d(s)
        color = Rgba(1, 0.72, 0, 1) if col else Rgba(0.24, 1, 0, 1)


        self.diagram.Publish(self.diagram_context)
        self.visualize_planes(idx_list)
        #don't change this order
        if self.do_viz_2:
            self.meshcat2.SetObject(f"/s",
                                    Sphere(0.05),
                                    color)
            self.meshcat2.SetTransform(f"/s",
                                       RigidTransform(RotationMatrix(),
                                                      s))


    def showres_s(self, s):
        q = self.forward_kin.ComputeQValue(s, self.q_star)
        self.showres(q)

    def transform(self, a, b, p1, p2, plane_verts):
        alpha = (-b - a.T @ p1) / (a.T @ (p2 - p1))
        offset = alpha * (p2 - p1) + p1
        z = np.array([0, 0, 1])
        crossprod = np.cross(viz_utils.normalize(a), z)
        if np.linalg.norm(crossprod) <= 1e-4:
            R = np.eye(3)
        else:
            ang = np.arcsin(np.linalg.norm(crossprod))
            axis = viz_utils.normalize(crossprod)
            # R = viz_utils.get_rotation_matrix(-axis, -ang)
            R = viz_utils.get_rotation_matrix(axis, ang)

        verts_tf = (R @ plane_verts.T).T + offset
        return verts_tf, RigidTransform(RotationMatrix(R), offset)

    def animate_s(self, traj, steps, runtime, idx_list = None, sleep_time = 0.1):
        # loop
        idx = 0
        going_fwd = True
        time_points = np.linspace(0, traj.end_time(), steps)

        for _ in range(runtime):
            # print(idx)
            t0 = time.time()
            q = self.forward_kin.ComputeQValue(traj.value(time_points[idx]), self.q_star)
            self.showres(q, idx_list)
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
            pause = sleep_time- (t1-t0)
            if pause > 0:
                time.sleep(pause)

    def _is_collision_pair_of_interest(self, idA, idB):
        for gid_pairs in self.collision_pairs_of_interest:
            if (idA, idB) == gid_pairs or (idB, idA) == gid_pairs:
                return True
        return False

    def visualize_planes(self, idx_list = None):
        idx_list = [i for i in range(len(self.certified_region_solution_list))] if idx_list is None else idx_list
        q = self.plant.GetPositions(self.plant_context)
        s = self.forward_kin.ComputeTValue(np.array(q), self.q_star)
        if self.color_dict is None:
            colors = viz_utils.n_colors(len(self.certified_region_solution_list), rgbs_ret=True)
            color_dict = {i: tuple(val / 255 for val in c) for i, c in enumerate(colors)}
        else:
            color_dict = self.color_dict

        color_ctr = 0
        for region_number, sol in enumerate(self.certified_region_solution_list):
            if region_number in idx_list:
                # point is in region so see plot interesting planes
                if np.all(sol.C @ s <= sol.d):
                    for i, plane in enumerate(self._region_to_planes_of_interest_dict[sol]):
                        idA = plane.positive_side_geometry.id() if plane.positive_side_geometry is not None else None
                        idB = plane.negative_side_geometry.id() if plane.negative_side_geometry is not None else None
                        if self._is_collision_pair_of_interest(idA, idB):
                            self._plot_plane(plane, color_dict[region_number], color_dict[region_number],
                                        color_dict[region_number], s,
                                        region_number)
                            color_ctr += 1
                else:
                    # exited region so remove the visualization associated to this solution
                    self.meshcat1.Delete(f"/planes/region{region_number}")
            else:
                # exited region so remove the visualization associated to this solution
                self.meshcat1.Delete(f"/planes/region{region_number}")

    def _plot_plane(self, plane, bodyA_color, bodyB_color, plane_color, s, region_number):
        # get the vertices of the separated bodies
        vert_A = GetVertices(plane.positive_side_geometry.geometry())
        dims_A = np.max(np.abs(vert_A[:, :-1] - vert_A[:, 1:]), axis=1)
        vert_B = GetVertices(plane.positive_side_geometry.geometry())
        dims_B = np.max(np.abs(vert_B[:, :-1] - vert_B[:, 1:]), axis=1)

        # get the geometry id of the separated bodies
        geomA = plane.positive_side_geometry.id()
        geomB = plane.negative_side_geometry.id()

        # get the equation of the plane
        b = plane.b
        a = plane.a
        b_eval = b.Evaluate(dict(zip(b.GetVariables(), s)))
        a_eval = np.array([a_idx.Evaluate(dict(zip(a_idx.GetVariables(), s))) for a_idx in a])

        # transform from expressed frame of plane to world frame
        X_EW = self.plant.GetBodyFromFrameId(
            self.plant.GetBodyFrameIdIfExists(plane.expressed_link)) \
            .body_frame().CalcPoseInWorld(self.plant_context).inverse()
        X_WE = X_EW.inverse()

        # transform vertices of body A expressed in body A into world frame
        X_WA = self.plant.GetBodyFromFrameId(
            self.plant.GetBodyFrameIdIfExists(plane.positive_side_geometry.body_index())) \
            .body_frame().CalcPoseInWorld(self.plant_context)
        vert_A = X_WA @ vert_A

        # transform vertices of body A expressed in body B into world frame
        X_WB = self.plant.GetBodyFromFrameId(
            self.plant.GetBodyFrameIdIfExists(plane.negative_side_geometry.body_index())) \
            .body_frame().CalcPoseInWorld(self.plant_context)
        vert_B = X_WB @ vert_B

        verts_tf_E, trans = self.transform(a_eval, b_eval, X_EW @ vert_A[:, 0],
                                           X_EW @ vert_B[:, 0], self.plane_verts)

        box_transform = X_WE @ trans
        # verts_tf = (X_WE @ verts_tf_E.T).T
        # verts_tf = np.vstack([verts_tf, verts_tf[::-1,:]])
        prefix = f"/planes/region{region_number}"

        def plot_polytope_highlight(id, dims, trans, color):
            box = Box(dims[0], dims[1], dims[2])
            name = prefix + f"/body{id.get_value()}"
            self.meshcat1.SetObject(name,
                                    box,
                                    Rgba(*color, 1))

            offset = np.array([0, 0, dims[2] / 2])
            t_final = trans @ RigidTransform(RotationMatrix(), offset)
            self.meshcat1.SetTransform(name, t_final)

        plot_polytope_highlight(geomA, dims_A, X_WA, bodyA_color)
        plot_polytope_highlight(geomB, dims_B, X_WB, bodyB_color)

        path = prefix + f"/plane/{geomA.get_value()}, {geomB.get_value()}"
        self.meshcat1.SetObject(path,
                                self.box,
                                Rgba(*plane_color, 0.7))
        self.meshcat1.SetTransform(path, box_transform)

    def draw_traj_s_space(self, traj, maxit):
        # evals end twice fix later
        for it in range(maxit):
            pt = np.append(traj.value(it * traj.end_time() / maxit),0)
            pt_nxt = np.append(traj.value((it + 1) * traj.end_time() / maxit),0)

            path = f"/traj/points{it}"
            self.meshcat2.SetLine(path, np.hstack([pt[:,np.newaxis], pt_nxt[:, np.newaxis]]),
                                 line_width = 2, rgba = Rgba(0.0, 0.0, 1, 1))

    def MakeFromHPolyhedronSceneGraph(self, query, geom, expressed_in=None):
        shape = query.inspector().GetShape(geom)
        if isinstance(shape, (Sphere, Ellipsoid)):
            raise ValueError(f"Sphere or Ellipsoid not Supported")
        return HPolyhedron(query, geom, expressed_in)

    def MakeFromVPolytopeSceneGraph(self, query, geom, expressed_in=None):
        shape = query.inspector().GetShape(geom)
        if isinstance(shape, (Sphere, Ellipsoid)):
            raise ValueError(f"Sphere or Ellipsoid not Supported")
        return VPolytope(query, geom, expressed_in)
