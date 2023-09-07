import numpy as np
from functools import partial
import visualization_utils as vis_utils
from iris_plant_visualizer import IrisPlantVisualizer
import ipywidgets as widgets
from IPython.display import display
from scipy.linalg import block_diag
import matplotlib.pyplot as plt
from pathlib import Path
import os

from pydrake.common import FindResourceOrThrow
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.systems.framework import DiagramBuilder
from pydrake.geometry import Role, GeometrySet, CollisionFilterDeclaration
from pydrake.all import RigidTransform, RollPitchYaw, RevoluteJoint
from pydrake.all import RotationMatrix, MeshcatVisualizer, StartMeshcat
import pydrake.symbolic as sym
from pydrake.solvers import MosekSolver, CommonSolverOption, SolverOptions, ScsSolver
from pydrake.all import (
    PointCloud,
    MeshcatVisualizerParams,
    Role,
    HalfSpace,
    CoulombFriction,
    Box,
    Rgba,
    WeldJoint,
    GeometrySet,
)
import pickle
from pydrake.polynomial import Polynomial as PolynomialCommon
import time
from scipy.spatial.transform import Rotation as sp_rot
from sampling_based_motion_planners import StraightLineCollisionChecker, PRMFixedEdges


from pydrake.all import (
    RationalForwardKinematics,
    ModelInstanceIndex,
    SpatialInertia,
    RevoluteJoint,
    FixedOffsetFrame,
    MultibodyPlant,
)
from pydrake.geometry.optimization import HPolyhedron, Hyperellipsoid
from pydrake.geometry.optimization_dev import CspaceFreePath


def AddPendulumLink(
    plant,
    pendulum_model_instance,
    link_dimensions,
    parent_frame,
    link_index: int,
    color=np.array([1, 0, 0, 1]),
):
    pendulum_box = Box(*link_dimensions)
    pendulum_inertia = SpatialInertia.SolidBoxWithDensity(1.0, *link_dimensions)

    link_name = f"link_{link_index}"
    pend_body = plant.AddRigidBody(link_name, pendulum_model_instance, pendulum_inertia)
    parent_to_child_attach_frame = plant.AddFrame(
        FixedOffsetFrame(
            f"{link_name}_joint_frame",
            pend_body.body_frame(),
            RigidTransform(p=(0, link_dimensions[1] / 2, 0)),
        )
    )
    next_frame = plant.AddFrame(
        FixedOffsetFrame(
            f"{link_name}_joint_frame",
            pend_body.body_frame(),
            RigidTransform(p=(0, -link_dimensions[1] / 2, 0)),
        )
    )
    # add the revolute joint,
    joint_lim = np.pi
    plant.AddJoint(
        RevoluteJoint(
            f"joint_{link_index}",
            parent_frame,
            parent_to_child_attach_frame,
            [0, 0, 1],
            -joint_lim,
            joint_lim,
        )
    )

    geom_id = plant.RegisterCollisionGeometry(
        pend_body, RigidTransform(), pendulum_box, link_name, CoulombFriction()
    )
    plant.RegisterVisualGeometry(
        pend_body, RigidTransform(), pendulum_box, link_name, color
    )

    return next_frame, geom_id


def N_Link_Pendulum(
    n: int,
    plant: MultibodyPlant,
    pendulum_model_instance: ModelInstanceIndex,
    link_dimensions: np.ndarray,
):
    assert n > 0
    parent_frame = plant.world_frame()
    colors = np.array(vis_utils.n_colors(n)) / 255
    colors = np.hstack([colors, np.ones((colors.shape[0], 1))])
    geom_ids = []
    for i in range(n):
        parent_frame, geom_id = AddPendulumLink(
            plant,
            pendulum_model_instance,
            link_dimensions,
            parent_frame,
            i + 1,
            colors[i],
        )
        geom_ids.append(geom_id)

    return geom_ids


def AddRandomBox(
    plant: MultibodyPlant,
    obstacle_model_instance: ModelInstanceIndex,
    index: int,
    pos_limits,
    size_limits,
    color=[0, 0, 0, 1],
):
    link_name = f"obstacle_{index}"

    center = np.append(np.random.uniform(pos_limits[0], pos_limits[1], 2), 0)

    random_rp = np.zeros(3)
    lim = np.pi / 10
    random_rp[2] = np.random.uniform(-lim, lim)
    angle = RotationMatrix(RollPitchYaw(random_rp))

    pose = RigidTransform(p=center) @ RigidTransform(R=angle)

    dimensions = np.append(np.random.uniform(size_limits[0], size_limits[1], 2), 0.001)
    box = Box(*dimensions)
    inertia = SpatialInertia.SolidBoxWithDensity(1, *dimensions)

    body = plant.AddRigidBody(link_name, obstacle_model_instance, inertia)
    plant.AddJoint(WeldJoint(link_name, plant.world_frame(), body.body_frame(), pose))
    plant.RegisterCollisionGeometry(
        body, RigidTransform(), box, link_name, CoulombFriction()
    )
    plant.RegisterVisualGeometry(body, RigidTransform(), box, link_name, color)


def AddNRandomBoxes(
    n: int,
    plant: MultibodyPlant,
    obstacle_model_instance: ModelInstanceIndex,
    pos_limits=(-10, 10),
    size_limits=(0.1, 1),
    color=[0, 0, 0, 1],
):
    assert n > 0
    for i in range(n):
        AddRandomBox(plant, obstacle_model_instance, i, pos_limits, size_limits, color)


def make_line_polys(plant: MultibodyPlant, prm):
    polys = np.empty(shape=(plant.num_positions(), len(prm.prm.edges())), dtype=object)
    for i, (s0, s1) in enumerate(prm.prm.edges()):
        for j in range(plant.num_positions()):
            polys[j, i] = PolynomialCommon(np.array([s0[j], s1[j] - s0[j]]))
    return polys


def build_n_link_k_boxes_plant_and_certifier(
    n: int, k: int, exclude_pend_self_collisions=True, plane_order=1
):
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)

    link_dimensions = (0.1, 0.2, 0.001)
    pendulum_model_instance = plant.AddModelInstance("pendulum")
    obstacle_model_instance = plant.AddModelInstance("obstacle")

    num_pend = n
    num_obstacles = k
    np.random.seed(num_pend * num_obstacles)

    pend_geom_ids = N_Link_Pendulum(
        num_pend, plant, pendulum_model_instance, np.arry(link_dimensions)
    )

    obstacle_pos_limits = link_dimensions[1] * num_pend * np.array([-1, 1])
    obstacle_size_limits = (0.01 / np.log(num_obstacles), 0.5 / np.log(num_obstacles))
    AddNRandomBoxes(
        num_obstacles,
        plant,
        obstacle_model_instance,
        obstacle_pos_limits,
        obstacle_size_limits,
    )

    plant.Finalize()
    if exclude_pend_self_collisions:
        collision_filter_manager = scene_graph.collision_filter_manager()
        pend_geom_set = GeometrySet(pend_geom_ids)
        decl = CollisionFilterDeclaration().ExcludeWithin(pend_geom_set)
        collision_filter_manager.Apply(decl)

    diagram = builder.Build()

    q_star = np.zeros(plant.num_positions())
    t0 = time.time()
    cspace_free_path = CspaceFreePath(
        plant, scene_graph, q_star, maximum_path_degree=1, plane_order=plane_order
    )
    t1 = time.time()
    print(f"Time to construct line certifier for {n}-links, {k}-boxes = {t1 - t0}s")
    return plant, scene_graph, diagram, cspace_free_path


def n_link_k_boxes_l_edges_experiement(
    n: int,
    k: int,
    l: list[int],
    file_save_name: str,
    exclude_pend_self_collisions=True,
    plane_order=1,
    good_collision_checker=True,
):
    (
        plant,
        scene_graph,
        diagram,
        cspace_free_path,
    ) = build_n_link_k_boxes_plant_and_certifier(
        n, k, exclude_pend_self_collisions, plane_order
    )
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)

    link_dimensions = (0.1, 0.2, 0.001)
    pendulum_model_instance = plant.AddModelInstance("pendulum")
    obstacle_model_instance = plant.AddModelInstance("obstacle")

    num_pend = n
    num_obstacles = k
    np.random.seed(num_pend * num_obstacles)

    pend_geom_ids = N_Link_Pendulum(num_pend)

    obstacle_pos_limits = link_dimensions[1] * num_pend * np.array([-1, 1])
    obstacle_size_limits = (0.01 / np.log(num_obstacles), 0.5 / np.log(num_obstacles))
    AddNRandomBoxes(num_obstacles, obstacle_pos_limits, obstacle_size_limits)

    plant.Finalize()
    if exclude_pend_self_collisions:
        collision_filter_manager = scene_graph.collision_filter_manager()
        pend_geom_set = GeometrySet(pend_geom_ids)
        decl = CollisionFilterDeclaration().ExcludeWithin(pend_geom_set)
        collision_filter_manager.Apply(decl)

    diagram = builder.Build()
    t0 = time.time()
    cspace_free_path = CspaceFreePath(
        plant, scene_graph, q_star, maximum_path_degree=1, plane_order=plane_order
    )
    t1 = time.time()

    q_star = np.zeros(plant.num_positions())
    Ratfk = RationalForwardKinematics(plant)

    diagram_col_context = diagram.CreateDefaultContext()
    plant_col_context = diagram.GetMutableSubsystemContext(plant, diagram_col_context)
    scene_graph_col_context = diagram.GetMutableSubsystemContext(
        scene_graph, diagram_col_context
    )
    query_port = scene_graph.get_query_output_port()

    def check_collision_q_by_query(q):
        if np.all(q >= plant.GetPositionLowerLimits()) and np.all(
            q <= plant.GetPositionUpperLimits()
        ):
            plant.SetPositions(plant_col_context, q)
            query_object = query_port.Eval(scene_graph_col_context)
            return 1 if query_object.HasCollisions() else 0
        else:
            return 1

    def check_collision_s_by_query(s):
        s = np.array(s)
        q = Ratfk.ComputeQValue(s, q_star)
        return check_collision_q_by_query(q)

    if good_collision_checker:
        collision_checker = StraightLineCollisionChecker(
            check_collision_s_by_query, 100
        )
    else:
        collision_checker = StraightLineCollisionChecker(check_collision_s_by_query, 10)

    def sample_col_free_point():
        q = np.random.uniform(
            plant.GetPositionLowerLimits(), plant.GetPositionUpperLimits()
        )
        s = Ratfk.ComputeSValue(q, q_star)
        return s

    prm = PRMFixedEdges(sample_col_free_point, l, collision_checker, dist_thresh=100)
    assert len(prm.prm.edges()) == l

    prm_poly_paths = make_line_polys(prm)

    cert_options = CspaceFreePath.FindSeparationCertificateGivenPathOptions()
    cert_options.terminate_segment_certification_at_failure = False
    cert_options.num_threads = -1
    cert_options.verbose = False
    cert_options.solver_id = MosekSolver.id()
    cert_options.solver_options = SolverOptions()
    cert_options.terminate_path_certification_at_failure = False

    print("Beginning certification")
    t0 = time.time()
    statistics, cert_result = cspace_free_path.FindSeparationCertificateGivenPath(
        prm_poly_paths, set(), cert_options
    )
    t1 = time.time()
    print(f"Certification of safe PRM in {t1 - t0}s")

    with open(file_save_name, "wb") as f:
        pickle.dump(statistics, f)


if __name__ == "__main__":
    # num_links = np.arange(2, 13, 1)
    num_links = np.arange(2, 5, 1)
    num_boxes = np.array([10, 50, 100])
    num_edges = np.array([100, 500, 1000])

    LL, NN, KK = np.meshgrid(num_edges, num_links, num_boxes)
    # iterate smallest to largest over num_links, then num_boxes, then num_edges
    for i in range(NN.shape[0]):
        for j in range(NN.shape[1]):
            for k in range(NN.shape[2]):
                print(NN[i, j, k], KK[i, j, k], LL[i, j, k])

    NN, KK, LL = np.meshgrid(num_links, num_boxes, num_edges)
