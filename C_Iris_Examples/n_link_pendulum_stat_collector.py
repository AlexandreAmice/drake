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
from tqdm import tqdm

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
from joblib import Parallel, delayed


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

    dimensions = np.append(np.random.uniform(size_limits[0], size_limits[1], 2), 0.001)
    box = Box(*dimensions)
    inertia = SpatialInertia.SolidBoxWithDensity(1, *dimensions)

    body = plant.AddRigidBody(link_name, obstacle_model_instance, inertia)

    random_rp = np.zeros(3)
    lim = np.pi / 10
    random_rp[2] = np.random.uniform(-lim, lim)
    angle = RotationMatrix(RollPitchYaw(random_rp))
    pose = RigidTransform(p=center) @ RigidTransform(R=angle)
    origin_local = pose.inverse() @ RigidTransform()
    while np.all(
        [
            -dimensions[i] / 2 <= origin_local.translation()[i] < dimensions[i] / 2
            for i in range(3)
        ]
    ):
        random_rp[2] = np.random.uniform(-lim, lim)
        angle = RotationMatrix(RollPitchYaw(random_rp))
        pose = RigidTransform(p=center) @ RigidTransform(R=angle)
        origin_local = pose.inverse() @ RigidTransform()

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


def build_n_link_k_boxes_plant(n: int, k: int, exclude_pend_self_collisions=True):
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)

    link_dimensions = (0.1, 0.2, 0.001)
    pendulum_model_instance = plant.AddModelInstance("pendulum")
    obstacle_model_instance = plant.AddModelInstance("obstacle")

    num_pend = n
    num_obstacles = k
    np.random.seed(num_pend * num_obstacles)

    pend_geom_ids = N_Link_Pendulum(
        num_pend, plant, pendulum_model_instance, np.array(link_dimensions)
    )

    scale = 2
    obstacle_pos_limits = link_dimensions[1] * num_pend * np.array([-scale, scale])
    obstacle_size_limits = (0.01 / np.sqrt(num_obstacles), 0.5 / np.sqrt(num_obstacles))
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
    return plant, scene_graph, diagram


def build_and_save_prm(
    prm_save_name: Path,
    num_edges: list[int],
    node_sampling_fun,  # generate a random sample. It does not need to be collision free.
    straight_line_col_checker,
    dist_thresh=0.1,
    num_neighbors=5,
    max_it=int(1e4),
    initial_points=None,
):
    num_edges.sort()
    prm = PRMFixedEdges(
        node_sampling_fun,  # generate a random sample. It does not need to be collision free.
        num_edges[0],
        straight_line_col_checker,
        dist_thresh,
        num_neighbors,
        max_it,
        initial_points,
    )
    with open(str(prm_save_name) + (f"_{num_edges[0]}_edges.pkl"), "wb") as f:
        pickle.dump(prm, f)
    for k in num_edges[1:]:
        edges_to_add = k - len(prm.prm.edges())
        prm.add_k_edges(
            edges_to_add,
            node_sampling_fun,
            straight_line_col_checker,
            max_it,
            num_neighbors,
            dist_thresh,
        )
        with open(str(prm_save_name) + f"_{k}_edges.pkl", "wb") as f:
            pickle.dump(prm, f)


def generate_prm_for_n_k(
    n,
    k,
    num_edges: list[int],
    use_good_checker,
    save_folder,
    exclude_pend_self_collision=True,
):
    (plant, scene_graph, diagram) = build_n_link_k_boxes_plant(
        n, k, exclude_pend_self_collision
    )
    diagram_col_context = diagram.CreateDefaultContext()
    plant_col_context = diagram.GetMutableSubsystemContext(plant, diagram_col_context)
    scene_graph_col_context = diagram.GetMutableSubsystemContext(
        scene_graph, diagram_col_context
    )
    query_port = scene_graph.get_query_output_port()
    q_star = np.zeros(plant.num_positions())
    Ratfk = RationalForwardKinematics(plant)

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

    def sample_s_point():
        q = np.random.uniform(
            plant.GetPositionLowerLimits(), plant.GetPositionUpperLimits()
        )
        s = Ratfk.ComputeSValue(q, q_star)
        return s

    if use_good_checker:
        collision_checker = StraightLineCollisionChecker(
            check_collision_s_by_query, 100
        )
    else:
        collision_checker = StraightLineCollisionChecker(check_collision_s_by_query, 10)

    good_checker = "good_checker" if use_good_checker else "bad_checker"

    prm_save_name = save_folder / f"prm_{n}_links_{k}_boxes_{good_checker}"
    build_and_save_prm(
        prm_save_name,
        num_edges,
        sample_s_point,
        collision_checker,
        dist_thresh=100,
        num_neighbors=4,
        max_it=int(1e4),
    )
    return None


def n_link_k_boxes_l_edges_experiment(
    n: int,
    k: int,
    num_edges: list[int],
    path_to_prm_folder: Path,
    path_to_save_folder: Path,
    exclude_pend_self_collisions=True,
    plane_order=1,
    # True uses good collision checker, False uses bad, or give True and False to do both
    use_good_checker=True,
):
    (plant, scene_graph, diagram) = build_n_link_k_boxes_plant(
        n, k, exclude_pend_self_collisions
    )
    q_star = np.zeros(plant.num_positions())
    good_checker = "good_checker" if use_good_checker else "bad_checker"
    t0 = time.time()
    cspace_free_path = CspaceFreePath(
        plant,
        scene_graph,
        q_star,
        maximum_path_degree=1,
        plane_order=plane_order,
    )
    t1 = time.time()
    print(f"Time to build collision checker {n}_links, {k}_boxes = {t1-t0}")
    cert_options = CspaceFreePath.FindSeparationCertificateGivenPathOptions()
    cert_options.terminate_segment_certification_at_failure = False
    cert_options.num_threads = -1
    cert_options.verbose = False
    cert_options.solver_id = MosekSolver.id()
    cert_options.solver_options = SolverOptions()
    cert_options.terminate_path_certification_at_failure = False
    for l in num_edges:
        prm_save_name = (
            path_to_prm_folder / f"prm_{n}_links_{k}_boxes_{good_checker}_{l}_edges.pkl"
        )
        with open(prm_save_name, "rb") as f:
            prm = pickle.load(f)
        prm_poly_paths = make_line_polys(plant, prm)
        print(
            f"Beginning certification of {n}_links, {k}_boxes, {l}_edges, {good_checker}"
        )
        t0 = time.time()
        (
            statistics,
            cert_result,
        ) = cspace_free_path.FindSeparationCertificateGivenPath(
            prm_poly_paths, set(), cert_options
        )
        t1 = time.time()
        print(
            f"Certification of {n}_links, {k}_boxes, {l}_edges, {good_checker} PRM in {t1 - t0}s"
        )

        save_path = (
            path_to_save_folder
            / f"{n}_links_{k}_obstacles_{l}_edges_{good_checker}_collision_checker.pkl"
        )
        with open(save_path, "wb") as f:
            pickle.dump(statistics, f)


num_links = np.arange(2, 14, 1)
num_boxes = np.array([10, 50, 100])
num_edges = np.array([100, 500, 1000])

# num_links = np.arange(2, 4, 1)
# num_boxes = np.array([10, 50])
# num_edges = np.array([100, 500, 1000])

NN, KK = np.meshgrid(num_links, num_boxes, indexing="ij")
n_list, k_list = NN.flatten(), KK.flatten()

path_to_prm_save_folder = Path(
    "/home/amice/Documents/coding_projects/drake/C_Iris_Examples/n_link_pend_prm"
)
path_to_data_save_folder = Path(
    "/home/amice/Documents/coding_projects/drake/C_Iris_Examples/n_link_pend_data"
)

if __name__ == "__main__2":
    Parallel(n_jobs=-1, backend="multiprocessing")(
        delayed(generate_prm_for_n_k)(
            n, k, num_edges.copy(), True, path_to_prm_save_folder, True
        )
        for (n, k) in tqdm(zip(n_list, k_list))
    )
    Parallel(n_jobs=-1, backend="multiprocessing")(
        delayed(generate_prm_for_n_k)(
            n, k, num_edges.copy(), False, path_to_prm_save_folder, True
        )
        for (n, k) in tqdm(zip(n_list, k_list))
    )

if __name__ == "__main__":
    for n, k in tqdm(zip(n_list, k_list)):
        n_link_k_boxes_l_edges_experiment(
            n,
            k,
            num_edges.copy(),
            path_to_prm_save_folder,
            path_to_data_save_folder,
            True,
            1,
            True,
        )
    for n, k in tqdm(zip(n_list, k_list)):
        n_link_k_boxes_l_edges_experiment(
            n,
            k,
            num_edges.copy(),
            path_to_prm_save_folder,
            path_to_data_save_folder,
            True,
            1,
            False,
        )
