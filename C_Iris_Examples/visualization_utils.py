import numpy as np
from functools import partial
from pydrake.all import (
    MathematicalProgram,
    le,
    SnoptSolver,
    SurfaceTriangle,
    TriangleSurfaceMesh,
    VPolytope,
    HPolyhedron,
    Sphere,
    RigidTransform,
    RotationMatrix,
    Rgba,
    MultibodyPlant,
    RationalForwardKinematics,
    Meshcat,
    LeafContext,
    PointCloud,
    Diagram,
    Context,
    RigidBody,
    Trajectory,
)
import mcubes
import time
from scipy.spatial import ConvexHull
from scipy.linalg import block_diag
from fractions import Fraction
import itertools
import random
import colorsys
from pydrake.all import PiecewisePolynomial
from pydrake.polynomial import Polynomial as CommonPolynomial

from dataclasses import dataclass


@dataclass
class VisualizationBundle:
    diagram: Diagram
    diagram_context: Context
    plant: MultibodyPlant
    plant_context: LeafContext
    rational_forward_kinematics: RationalForwardKinematics
    meshcat_instance: Meshcat
    q_star: np.ndarray


@dataclass
class TrajectoryVisualizationOptions:
    start_color: Rgba = Rgba(0, 1, 0, 1)
    start_size: float = 0.01
    end_color: Rgba = Rgba(1, 0, 0, 1)
    end_size: float = 0.01
    path_color: Rgba = Rgba(0, 0, 1, 1)
    path_size: float = 0.01
    num_points: int = 100


def plot_surface(
    meshcat_instance,
    path,
    X,
    Y,
    Z,
    rgba=Rgba(0.87, 0.6, 0.6, 1.0),
    wireframe=False,
    wireframe_line_width=1.0,
):
    # taken from
    # https://github.com/RussTedrake/manipulation/blob/346038d7fb3b18d439a88be6ed731c6bf19b43de/manipulation
    # /meshcat_cpp_utils.py#L415
    (rows, cols) = Z.shape
    assert np.array_equal(X.shape, Y.shape)
    assert np.array_equal(X.shape, Z.shape)

    vertices = np.empty((rows * cols, 3), dtype=np.float32)
    vertices[:, 0] = X.reshape((-1))
    vertices[:, 1] = Y.reshape((-1))
    vertices[:, 2] = Z.reshape((-1))

    # Vectorized faces code from https://stackoverflow.com/questions/44934631/making-grid-triangular-mesh-quickly
    # -with-numpy  # noqa
    faces = np.empty((rows - 1, cols - 1, 2, 3), dtype=np.uint32)
    r = np.arange(rows * cols).reshape(rows, cols)
    faces[:, :, 0, 0] = r[:-1, :-1]
    faces[:, :, 1, 0] = r[:-1, 1:]
    faces[:, :, 0, 1] = r[:-1, 1:]
    faces[:, :, 1, 1] = r[1:, 1:]
    faces[:, :, :, 2] = r[1:, :-1, None]
    faces.shape = (-1, 3)

    # TODO(Russ): support per vertex / Colormap colors.
    meshcat_instance.SetTriangleMesh(
        path, vertices.T, faces.T, rgba, wireframe, wireframe_line_width
    )


def plot_point(point, meshcat_instance, name, color=Rgba(0.06, 0.0, 0, 1), radius=0.01):
    meshcat_instance.SetObject(name, Sphere(radius), color)
    meshcat_instance.SetTransform(
        name, RigidTransform(RotationMatrix(), stretch_array_to_3d(point))
    )


def plot_polytope(
    polytope,
    meshcat_instance,
    name,
    resolution=50,
    color=None,
    wireframe=True,
    random_color_opacity=0.2,
    fill=True,
    line_width=10,
):
    if color is None:
        color = Rgba(*np.random.rand(3), random_color_opacity)
    if polytope.ambient_dimension == 3:
        verts, triangles = get_plot_poly_mesh(polytope, resolution=resolution)
        meshcat_instance.SetObject(
            name, TriangleSurfaceMesh(triangles, verts), color, wireframe=wireframe
        )

    else:
        plot_hpoly2d(
            polytope,
            meshcat_instance,
            name,
            color,
            line_width=line_width,
            fill=fill,
            resolution=resolution,
            wireframe=wireframe,
        )


def plot_hpoly2d(
    polytope,
    meshcat_instance,
    name,
    color,
    line_width=8,
    fill=False,
    resolution=30,
    wireframe=True,
):
    # plot boundary
    vpoly = VPolytope(polytope)
    verts = vpoly.vertices()
    hull = ConvexHull(verts.T)
    inds = np.append(hull.vertices, hull.vertices[0])
    hull_drake = verts.T[inds, :].T
    hull_drake3d = np.vstack([hull_drake, np.zeros(hull_drake.shape[1])])
    color_RGB = Rgba(color.r(), color.g(), color.b(), 1)
    meshcat_instance.SetLine(name, hull_drake3d, line_width=line_width, rgba=color_RGB)
    if fill:
        width = 0.5
        C = block_diag(polytope.A(), np.array([-1, 1])[:, np.newaxis])
        d = np.append(polytope.b(), width * np.ones(2))
        hpoly_3d = HPolyhedron(C, d)
        verts, triangles = get_plot_poly_mesh(hpoly_3d, resolution=resolution)
        meshcat_instance.SetObject(
            name + "/fill",
            TriangleSurfaceMesh(triangles, verts),
            color,
            wireframe=wireframe,
        )


def get_plot_poly_mesh(polytope, resolution):
    def inpolycheck(q0, q1, q2, A, b):
        q = np.array([q0, q1, q2])
        res = np.min(1.0 * (A @ q - b <= 0))
        return res

    aabb_max, aabb_min = get_AABB_limits(polytope)

    col_hand = partial(inpolycheck, A=polytope.A(), b=polytope.b())
    vertices, triangles = mcubes.marching_cubes_func(
        tuple(aabb_min),
        tuple(aabb_max),
        resolution,
        resolution,
        resolution,
        col_hand,
        0.5,
    )
    tri_drake = [SurfaceTriangle(*t) for t in triangles]
    return vertices, tri_drake


def get_AABB_limits(hpoly, dim=3):
    max_limits = []
    min_limits = []
    A = hpoly.A()
    b = hpoly.b()

    for idx in range(dim):
        aabbprog = MathematicalProgram()
        x = aabbprog.NewContinuousVariables(dim, "x")
        cost = x[idx]
        aabbprog.AddCost(cost)
        aabbprog.AddConstraint(le(A @ x, b))
        solver = SnoptSolver()
        result = solver.Solve(aabbprog)
        min_limits.append(result.get_optimal_cost() - 0.01)
        aabbprog = MathematicalProgram()
        x = aabbprog.NewContinuousVariables(dim, "x")
        cost = -x[idx]
        aabbprog.AddCost(cost)
        aabbprog.AddConstraint(le(A @ x, b))
        solver = SnoptSolver()
        result = solver.Solve(aabbprog)
        max_limits.append(-result.get_optimal_cost() + 0.01)
    return max_limits, min_limits


def stretch_array_to_3d(arr, val=0.0):
    if arr.shape[0] < 3:
        arr = np.append(arr, val * np.ones((3 - arr.shape[0])))
    return arr


def infinite_hues():
    yield Fraction(0)
    for k in itertools.count():
        i = 2**k  # zenos_dichotomy
        for j in range(1, i, 2):
            yield Fraction(j, i)


def hue_to_hsvs(h: Fraction):
    # tweak values to adjust scheme
    for s in [Fraction(6, 10)]:
        for v in [Fraction(6, 10), Fraction(9, 10)]:
            yield (h, s, v)


def rgb_to_css(rgb) -> str:
    uint8tuple = map(lambda y: int(y * 255), rgb)
    return tuple(uint8tuple)


def css_to_html(css):
    return f"<text style=background-color:{css}>&nbsp;&nbsp;&nbsp;&nbsp;</text>"


def n_colors(n=33, rgbs_ret=False):
    hues = infinite_hues()
    hsvs = itertools.chain.from_iterable(hue_to_hsvs(hue) for hue in hues)
    rgbs = (colorsys.hsv_to_rgb(*hsv) for hsv in hsvs)
    csss = (rgb_to_css(rgb) for rgb in rgbs)
    to_ret = (
        list(itertools.islice(csss, n)) if rgbs_ret else list(itertools.islice(csss, n))
    )
    return to_ret


def draw_traj(
    meshcat_instance,
    traj,
    maxit,
    name="/trajectory",
    color=Rgba(0, 0, 0, 1),
    line_width=3,
):
    pts = np.squeeze(
        np.array([traj.value(it * traj.end_time() / maxit) for it in range(maxit)])
    )
    pts_3d = np.hstack([pts, 0 * np.ones((pts.shape[0], 3 - pts.shape[1]))]).T
    meshcat_instance.SetLine(name, pts_3d, line_width, color)


def generate_walk_around_polytope(h_polytope, num_verts):
    v_polytope = VPolytope(h_polytope)
    verts_to_visit_index = np.random.randint(
        0, v_polytope.vertices().shape[1], num_verts
    )
    verts_to_visit = v_polytope.vertices()[:, verts_to_visit_index]
    t_knots = np.linspace(0, 1, verts_to_visit.shape[1])
    lin_traj = PiecewisePolynomial.FirstOrderHold(t_knots, verts_to_visit)
    return lin_traj


def visualize_s_space_trajectory(
    vis_bundle: VisualizationBundle,
    trajectory: Trajectory,
    body: RigidBody,
    path: str,
    options=TrajectoryVisualizationOptions(),
):
    points = []
    for t in np.linspace(
        trajectory.start_time(), trajectory.end_time(), options.num_points
    ):
        s = trajectory.value(t)
        q = vis_bundle.rational_forward_kinematics.ComputeQValue(s, vis_bundle.q_star)
        vis_bundle.plant.SetPositions(vis_bundle.plant_context, q)
        points.append(
            vis_bundle.plant.EvalBodyPoseInWorld(
                vis_bundle.plant_context, body
            ).translation()
        )
    points = np.array(points)
    pc = PointCloud(len(points))
    pc.mutable_xyzs()[:] = points.T
    pc.mutable_xyzs()[:, 2] = 1
    vis_bundle.meshcat_instance.SetObject(
        path, pc, point_size=options.path_size, rgba=options.path_color
    )

    start_s = trajectory.value(trajectory.start_time())
    start_q = vis_bundle.rational_forward_kinematics.ComputeQValue(
        start_s, vis_bundle.q_star
    )
    vis_bundle.plant.SetPositions(vis_bundle.plant_context, start_q)
    plot_point(
        vis_bundle.plant.EvalBodyPoseInWorld(
            vis_bundle.plant_context, body
        ).translation(),
        vis_bundle.meshcat_instance,
        path + "/start",
        options.start_color,
        options.start_size,
    )

    end_s = trajectory.value(trajectory.end_time())
    end_q = vis_bundle.rational_forward_kinematics.ComputeQValue(
        end_s, vis_bundle.q_star
    )
    vis_bundle.plant.SetPositions(vis_bundle.plant_context, end_q)
    plot_point(
        vis_bundle.plant.EvalBodyPoseInWorld(
            vis_bundle.plant_context, body
        ).translation(),
        vis_bundle.meshcat_instance,
        path + "/end",
        options.end_color,
        options.end_size,
    )


def visualize_s_space_segment(
    vis_bundle: VisualizationBundle,
    segment_start: np.ndarray,
    segment_end: np.ndarray,
    body: RigidBody,
    path: str,
    options=TrajectoryVisualizationOptions(),
):
    poly_traj = np.array(
        [
            CommonPolynomial([segment_start[i], segment_end[i] - segment_start[i]])
            for i in range(segment_start.shape[0])
        ]
    ).reshape(1, -1)
    trajectory = PiecewisePolynomial(poly_traj, [0, 1])
    visualize_s_space_trajectory(vis_bundle, trajectory, body, path, options)


def visualize_body_at_s(
    vis_bundle: VisualizationBundle, body, s, name, point_rad, color
):
    q = vis_bundle.rational_forward_kinematics.ComputeQValue(s, vis_bundle.q_star)
    vis_bundle.plant.SetPositions(vis_bundle.plant_context, q)
    task_space_point = vis_bundle.plant.EvalBodyPoseInWorld(
        vis_bundle.plant_context, body
    ).translation()
    vis_bundle.meshcat_instance.SetObject(name, Sphere(point_rad), color)
    vis_bundle.meshcat_instance.SetTransform(name, RigidTransform(p=task_space_point))


def animate_traj_s(vis_bundle, traj, steps, runtime, idx_list=None, sleep_time=0.1):
    # loop
    idx = 0
    going_fwd = True
    time_points = np.linspace(0, traj.end_time(), steps)
    frame_count = 0
    for _ in range(runtime):
        # print(idx)
        t0 = time.time()
        s = traj.value(time_points[idx])
        q = vis_bundle.rational_forward_kinematics.ComputeQValue(s, vis_bundle.q_star)
        vis_bundle.plant.SetPositions(vis_bundle.plant_context, q)
        vis_bundle.diagram.SetTime(frame_count * 0.01)
        vis_bundle.diagram.ForcedPublish(vis_bundle.task_space_diagram_context)
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
