import numpy as np
import matplotlib.pyplot as plt
import meshcat
from pydrake.all import (MathematicalProgram, le, SnoptSolver)
from functools import partial
import mcubes
from pydrake.all import RotationMatrix, RigidTransform
import colorsys
import itertools
from fractions import Fraction
import random


def infinite_hues():
    yield Fraction(0)
    for k in itertools.count():
        i = 2**k # zenos_dichotomy
        for j in range(1,i,2):
            yield Fraction(j,i)

def hue_to_hsvs(h: Fraction):
    # tweak values to adjust scheme
    for s in [Fraction(6,10)]:
        for v in [Fraction(6,10), Fraction(9,10)]:
            yield (h, s, v)

def rgb_to_css(rgb) -> str:
    uint8tuple = map(lambda y: int(y*255), rgb)
    return tuple(uint8tuple)

def css_to_html(css):
    return f"<text style=background-color:{css}>&nbsp;&nbsp;&nbsp;&nbsp;</text>"

def n_colors(n=33, rgbs_ret = False):
    hues = infinite_hues()
    hsvs = itertools.chain.from_iterable(hue_to_hsvs(hue) for hue in hues)
    rgbs = (colorsys.hsv_to_rgb(*hsv) for hsv in hsvs)
    csss = (rgb_to_css(rgb) for rgb in rgbs)
    to_ret = list(itertools.islice(csss, n)) if rgbs_ret else list(itertools.islice(csss, n))
    return to_ret

def n_colors_random(n=33, rgbs_ret = False):
    colors = n_colors(100 * n, rgbs_ret)
    return random.sample(colors, n)

class PiecewiseLinearTrajectory:
    def __init__(self, path, duration):
        self.path = path
        self.duration = duration
        self.num_waypoints = len(self.path)

    def value(self, time):
        prog_frac = np.clip(time/self.duration, a_min = 0, a_max = 1-1e-5)*\
                    (self.num_waypoints-1)
        prog_int = int(prog_frac)
        prog_part = prog_frac-prog_int
        wp1 = self.path[prog_int]
        wp2 = self.path[prog_int+1]
        return wp1 + prog_part*(wp2-wp1)

    def end_time(self,):
        return self.duration

def animate_trajectory(traj, publisher, steps, runtime):
    #loop
    idx = 0
    going_fwd = True
    time_points = np.linspace(0, traj.end_time(), steps) 

    for _ in range(runtime):
        q = traj.value(time_points[idx])
        publisher(q.reshape(-1,))
        if going_fwd:
            if idx + 1 < steps:
                idx += 1
            else:
                going_fwd = False
                idx -=1
        else:
            if idx-1 >= 0:
                idx -=1
            else:
                going_fwd = True
                idx +=1

def animate_t(traj, publisher, steps, t_to_q, runtime):
    #loop
    idx = 0
    going_fwd = True
    time_points = np.linspace(0, traj.end_time(), steps) 

    for _ in range(runtime):
        #print(idx)
        q = t_to_q(traj.value(time_points[idx]).reshape(1, -1)).squeeze()
        publisher(q.reshape(-1,))
        if going_fwd:
            if idx + 1 < steps:
                idx += 1
            else:
                going_fwd = False
                idx -=1
        else:
            if idx-1 >= 0:
                idx -=1
            else:
                going_fwd = True
                idx +=1


def plot(traj, steps, runtime):
    #loop
    idx = 0
    going_fwd = True
    time_points = np.linspace(0, traj.end_time(), steps) 
    traj_list = []
    
    for _ in range(runtime):
        #print(idx)
        traj_list.append(traj.value(time_points[idx]).reshape(-1,))
        if going_fwd:
            if idx + 1 < steps:
                idx += 1
            else:
                going_fwd = False
                idx -=1
        else:
            if idx-1 >= 0:
                idx -=1
            else:
                going_fwd = True
                idx +=1
    
    traj_arr = np.array(traj_list)

    fig, ax = plt.subplots(1,1, figsize=(10,6), dpi=72*3)
    data_dims = traj_arr.shape[1]
    for joint_idx in range(data_dims):
        ax.plot(np.arange(len(traj_arr[:,joint_idx])),traj_arr[:,joint_idx], label=f'Joint {joint_idx+1}')
    ax.legend(loc='upper center', ncol=data_dims)
    ax.set_ylim([-np.pi, np.pi])
    plt.show()

def meshcat_line(x_start, x_end, width):
    x_end_shift = x_end.copy()
    x_end_shift[0:2] += width
    x_end_shift2 = x_end.copy()
    x_end_shift2[0:1] += 0.5*width
    x_start_shift = x_start.copy()
    x_start_shift[1:2] += width
    x_start_shift2 = x_start.copy()
    x_start_shift2[0:1] += 0.5*width

    points = np.array([[x_start, x_end, x_end_shift, x_start_shift, x_start_shift2, x_end_shift2]]).reshape(-1,3)
    triangles = np.array([[0,1,2],[0,2,3],[0,1,3],[1,2,3], [0,4,5], [0,3,5],]).reshape(-1,3)
    mc_geom = meshcat.geometry.TriangularMeshGeometry(points, triangles)
    return mc_geom

def rgb_to_hex(rgb):
    return '0x%02x%02x%02x' % rgb

def get_AABB_limits(hpoly, dim = 3):
    max_limits = []
    min_limits = []
    A = hpoly.A()
    b = hpoly.b()

    for idx in range(dim):
        aabbprog = MathematicalProgram()
        x = aabbprog.NewContinuousVariables(dim, 'x')
        cost = x[idx]
        aabbprog.AddCost(cost)
        aabbprog.AddConstraint(le(A@x,b))
        solver = SnoptSolver()
        result = solver.Solve(aabbprog)
        min_limits.append(result.get_optimal_cost()-0.01)
        aabbprog = MathematicalProgram()
        x = aabbprog.NewContinuousVariables(dim, 'x')
        cost = -x[idx]
        aabbprog.AddCost(cost)
        aabbprog.AddConstraint(le(A@x,b))
        solver = SnoptSolver()
        result = solver.Solve(aabbprog)
        max_limits.append(-result.get_optimal_cost() + 0.01)
    return max_limits, min_limits

def plot_3d_poly(region, resolution, vis, name, mat = None, verbose = False):
    
    def inpolycheck(q0,q1,q2, A, b):
        q = np.array([q0, q1, q2])
        res = np.min(1.0*(A@q-b<=0))
        #print(res)
        return res
    
    aabb_max, aabb_min = get_AABB_limits(region)
    if verbose:
        print('AABB:', aabb_min, aabb_max)
    col_hand = partial(inpolycheck, A=region.A(), b=region.b())
    vertices, triangles = mcubes.marching_cubes_func(tuple(aabb_min), 
                                                     tuple(aabb_max),
                                                     resolution, 
                                                     resolution, 
                                                     resolution, 
                                                     col_hand, 
                                                     0.5)
    if mat is None:
        mat = meshcat.geometry.MeshLambertMaterial(color=0x000000 , wireframe=True)
        mat.opacity = 0.3
    vis[name].set_object(
            meshcat.geometry.TriangularMeshGeometry(vertices, triangles),
            mat)

def plot_point(loc, radius, mat, vis, marker_id):
    vis['markers'][marker_id].set_object(
                meshcat.geometry.Sphere(radius), mat)
    vis['markers'][marker_id].set_transform(
                meshcat.transformations.translation_matrix(loc))


def crossmat(vec): 
    R = np.zeros((3,3))
    R[0,1] = -vec[2]
    R[0,2] = vec[1]
    R[1,0] = vec[2]
    R[1,2] = -vec[0]
    R[2,0] = -vec[1]
    R[2,1] = vec[0]
    return R

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm
    
def get_rotation_matrix(axis, theta):
    R = np.cos(theta)*np.eye(3) + np.sin(theta)*crossmat(axis) + (1-np.cos(theta))*(axis.reshape(-1,1)@axis.reshape(-1,1).T)
    return R


def plot_regions(vis, regions, ellipses = None, region_suffix='', randomize_colors = False, wireframe = True):
    if randomize_colors:
        colors = n_colors_random(len(regions))
    else:
        colors = n_colors(len(regions))

    for i, region in enumerate(regions):
        c = colors[i]
        mat = meshcat.geometry.MeshLambertMaterial(color=rgb_to_hex(c), wireframe=wireframe)
        mat.opacity = 0.5
        plot_3d_poly(region=region,
                           resolution=30,
                           vis=vis['iris']['regions'+region_suffix],
                           name=str(i),
                           mat=mat,
                           wireframe = wireframe)
        if ellipses is not None:
            C = ellipses[i].A()  # [:, (0,2,1)]
            d = ellipses[i].center()  # [[0,2,1]]
            radii, R = np.linalg.eig(C.T @ C)
            R[:, 0] = R[:, 0] * np.linalg.det(R)
            Rot = RotationMatrix(R)

            transf = RigidTransform(Rot, d)
            mat = meshcat.geometry.MeshLambertMaterial(color=rgb_to_hex(c), wireframe=wireframe)
            mat.opacity = 0.15
            vis['iris']['ellipses'+region_suffix][str(i)].set_object(
                meshcat.geometry.Ellipsoid(np.divide(1, np.sqrt(radii))),
                mat)

            vis['iris']['ellipses'+region_suffix][str(i)].set_transform(transf.GetAsMatrix4())

    return vis



def stretch_array_to_3d(arr, val = 0.):
    if arr.shape[0] < 3:
        arr = np.append(arr, val*np.ones((3 - arr.shape[0])))
    return arr