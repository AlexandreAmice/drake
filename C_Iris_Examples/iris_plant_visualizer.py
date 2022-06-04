import numpy as np
import scipy
from meshcat.servers.zmqserver import start_zmq_server_as_subprocess
from meshcat import Visualizer
import meshcat
from pydrake.all import (ConnectMeshcatVisualizer, HPolyhedron,
                         VPolytope, Sphere, Ellipsoid, InverseKinematics,
                         RationalForwardKinematics, GeometrySet, Role)
from functools import partial
import mcubes
import C_Iris_Examples.visualizations_utils as viz_utils
import pydrake.symbolic as sym
from IPython.display import display
from scipy.spatial import Delaunay




class IrisPlantVisualizer:
    def __init__(self, plant, builder, scene_graph, **kwargs):
        proc, zmq_url, web_url = start_zmq_server_as_subprocess(server_args=[])
        proc2, zmq_url2, web_url2 = start_zmq_server_as_subprocess(server_args=[])
        self.vis = Visualizer(zmq_url=zmq_url)
        self.vis.delete()
        self.vis2 = Visualizer(zmq_url=zmq_url2)
        self.vis2.delete()

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
        self.q_upper_limits = plant.GetPositionUpperLimits()
        self.s_upper_limits = self.forward_kin.ComputeTValue(self.q_upper_limits, self.q_star)

        self.viz_role = kwargs.get('viz_role', Role.kIllustration)
        visualizer = ConnectMeshcatVisualizer(self.builder, scene_graph, zmq_url=zmq_url,
                                              delete_prefix_on_load=False, role=self.viz_role)
        self.diagram = self.builder.Build()
        visualizer.load()
        self.diagram_context = self.diagram.CreateDefaultContext()
        self.plant_context = plant.GetMyContextFromRoot(self.diagram_context)
        self.diagram.Publish(self.diagram_context)

        self.ik = InverseKinematics(plant, self.plant_context)
        self.collision_constraint = self.ik.AddMinimumDistanceConstraint(1e-4, 0.01)
        self.col_func_handle = partial(self.eval_cons, c=self.collision_constraint, tol=0.01)
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
        verts = []

        for idxx in range(len(x)):
            for idxy in range(len(y)):
                verts.append(np.array([x[idxx], y[idxy]]))
        self.tri = Delaunay(verts)
        self.plane_triangles = self.tri.simplices
        self.plane_verts = self.tri.points[:, :]
        self.plane_verts = np.concatenate((self.plane_verts, 0 * self.plane_verts[:, 0].reshape(-1, 1)), axis=1)


        #region -> (collision -> plane dictionary)
        self.region_to_collision_pair_to_plane_dictionary = None


        self.cube_tri = np.array([[0,2,1],[0,3,2],
                     [4,6,5],[4,7,6],
                     [4,3,7],[3,0,7],
                     [0,6,7],[0,1,6],
                     [1,5,6],[1,2,5],
                     [2,4,5],[2,3,4]])


        self._certified_region_solution_list = []
        self._collision_pairs_of_interest = []
        self._region_to_planes_of_interest_dict = {}

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
                A = plane.positive_side_polytope.get_id() if plane.positive_side_polytope is not None else None
                B = plane.negative_side_polytope.get_id() if plane.negative_side_polytope is not None else None
                for gid_pairs in self._collision_pairs_of_interest:
                    if (A, B) == gid_pairs or (B, A) == gid_pairs:
                        self._region_to_planes_of_interest_dict[certified_region_solution].append(plane)


    def jupyter_cell(self,):
        display(self.vis.jupyter_cell())
        display(self.vis2.jupyter_cell())

    def eval_cons(self, q, c, tol):
        return 1 - 1 * float(c.evaluator().CheckSatisfied(q, tol))

    def eval_cons_rational(self, *s):
        s = np.array(s)
        q = self.forward_kin.ComputeQValue(np.array(s), self.q_star)
        return self.col_func_handle(q)

    def visualize_collision_constraint(self, N = 50):
        """
        :param N: N is density of marchingcubes grid. Runtime scales cubically in N
        :return:
        """
        vertices, triangles = mcubes.marching_cubes_func(tuple(self.s_lower_limits),
                                                         tuple(self.s_upper_limits),
                                                         N, N, N, self.col_func_handle_rational, 0.5)
        self.vis2["collision_constraint"].set_object(
            meshcat.geometry.TriangularMeshGeometry(vertices, triangles),
            meshcat.geometry.MeshLambertMaterial(color=0xff0000, wireframe=True))

    def plot_regions(self, regions, ellipses = None, region_suffix = '', randomize_colors = False):
        viz_utils.plot_regions(self.vis2, regions, ellipses, region_suffix, randomize_colors)

    def plot_seedpoints(self, seed_points):
        for i in range(seed_points.shape[0]):
            self.vis2['iris']['seedpoints']["seedpoint"+str(i)].set_object(
                        meshcat.geometry.Sphere(0.05), meshcat.geometry.MeshLambertMaterial(color=0x0FB900))
            self.vis2['iris']['seedpoints']["seedpoint"+str(i)].set_transform(
                    meshcat.transformations.translation_matrix(seed_points[i,:]))

    def showres(self,q):
        self.plant.SetPositions(self.plant_context, q)
        col = self.col_func_handle(q)
        s = self.forward_kin.ComputeTValue(np.array(q), self.q_star)
        if col:
            self.vis2["s"].set_object(
                meshcat.geometry.Sphere(0.1), meshcat.geometry.MeshLambertMaterial(color=0xFFB900))
            self.vis2["s"].set_transform(
                meshcat.transformations.translation_matrix(s))
        else:
            self.vis2["s"].set_object(
                meshcat.geometry.Sphere(0.1), meshcat.geometry.MeshLambertMaterial(color=0x3EFF00))
            self.vis2["s"].set_transform(
                meshcat.transformations.translation_matrix(s))
        self.diagram.Publish(self.diagram_context)

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
            R = viz_utils.get_rotation_matrix(axis, -ang)

        verts_tf = (R @ plane_verts.T).T + offset
        return verts_tf

    def animate_s(self, traj, steps, runtime):
        # loop
        idx = 0
        going_fwd = True
        time_points = np.linspace(0, traj.end_time(), steps)

        for _ in range(runtime):
            # print(idx)
            q = self.forward_kin.ComputeQValue(traj.value(time_points[idx]), self.q_star)
            if self.region_to_collision_pair_to_plane_dictionary is not None:
                self.show_res_with_planes(q)
            else:
                self.showres(q)
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

    def _is_collision_pair_of_interest(self, idA, idB):
        for gid_pairs in self.collision_pairs_of_interest:
            if (idA, idB) == gid_pairs or (idB, idA) == gid_pairs:
                return True
        return False

    def visualize_planes(self):
        q = self.plant.GetPositions(self.plant_context)
        s = self.forward_kin.ComputeTValue(np.array(q), self.q_star)
        for region_number, sol in enumerate(self.certified_region_solution_list):

            # point is in region so see plot interesting planes
            if np.all(sol.C @ s <= sol.d):
                num_colors = len(sol.separating_planes)
                colors = viz_utils.n_colors(3 * num_colors)
                plane_colors = colors[:num_colors]
                body_colors = colors[num_colors:]

                for i, plane in enumerate(self._region_to_planes_of_interest_dict[sol]):
                    idA = plane.positive_side_polytope.get_id() if plane.positive_side_polytope is not None else None
                    idB = plane.negative_side_polytope.get_id() if plane.negative_side_polytope is not None else None
                    if self._is_collision_pair_of_interest(idA, idB):
                        self._plot_plane(plane, body_colors[2*i], body_colors[2*i+1], plane_colors[i], s,
                                         region_number)
            else:
                # exited region so remove the visualization associated to this solution
                self.vis[f"region{region_number}"].delete()


    def _delete_plane_from_viz(self, plane, region_number):
        geomA = plane.positive_side_polytope.get_id()
        geomB = plane.negative_side_polytope.get_id()
        self.vis[f"region{region_number}"][f"body{geomA.get_value()}"].delete()
        self.vis[f"region{region_number}"][f"body{geomB.get_value()}"].delete()
        self.vis[f"region{region_number}"]["plane"][f"{geomA.get_value()}, {geomB.get_value()}"].delete()

    def _plot_plane(self, plane, bodyA_color, bodyB_color, plane_color, s, region_number):
        # get the vertices of the separated bodies
        vert_A = plane.positive_side_polytope.p_BV()[:, :]
        vert_B = plane.negative_side_polytope.p_BV()[:, :]

        # get the geometry id of the separated bodies
        geomA = plane.positive_side_polytope.get_id()
        geomB = plane.negative_side_polytope.get_id()

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
            self.plant.GetBodyFrameIdIfExists(plane.positive_side_polytope.body_index())) \
            .body_frame().CalcPoseInWorld(self.plant_context)
        vert_A = X_WA @ vert_A

        # transform vertices of body A expressed in body B into world frame
        X_WB = self.plant.GetBodyFromFrameId(
            self.plant.GetBodyFrameIdIfExists(plane.negative_side_polytope.body_index())) \
            .body_frame().CalcPoseInWorld(self.plant_context)
        vert_B = X_WB @ vert_B

        verts_tf_E = self.transform(a_eval, b_eval, X_EW @ vert_A[:, 0],
                                    X_EW @ vert_B[:, 0], self.plane_verts)
        verts_tf = (X_WE @ verts_tf_E.T).T

        def plot_polytope_highlight(id, verts, color):
            mat = meshcat.geometry.MeshLambertMaterial(color=viz_utils.rgb_to_hex(color),
                                                       wireframe=False)
            mat.opacity = 1.
            self.vis[f"region{region_number}"][f"body{id.get_value()}"].set_object(
                meshcat.geometry.TriangularMeshGeometry(verts.T, self.cube_tri),
                mat)

        plot_polytope_highlight(geomA, vert_A, bodyA_color)
        plot_polytope_highlight(geomB, vert_B, bodyB_color)

        mat = meshcat.geometry.MeshLambertMaterial(color=plane_color,
                                                   wireframe=False)
        mat.opacity = 0.5
        self.vis[f"region{region_number}"]["plane"][f"{geomA.get_value()}, {geomB.get_value()}"].set_object(
            meshcat.geometry.TriangularMeshGeometry(verts_tf, self.plane_triangles),
            mat)

    def draw_traj_s_space(self, traj, maxit, name):
        # evals end twice fix later
        for it in range(maxit):
            pt = traj.value(it * traj.end_time() / maxit)
            pt_nxt = traj.value((it + 1) * traj.end_time() / maxit)

            mat = meshcat.geometry.MeshLambertMaterial(color=0xFFF812)
            mat.reflectivity = 1.0
            self.vis2[name]['traj']['points' + str(it)].set_object(viz_utils.meshcat_line(pt.squeeze(), pt_nxt.squeeze(), width=0.03),
                                                              mat)

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
