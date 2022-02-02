import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import meshcat
from pydrake.all import (MathematicalProgram, Variable, HPolyhedron, le, SnoptSolver, Solve) 
from functools import partial
import mcubes
from pydrake.all import RotationMatrix, RigidTransform
import colorsys
import cdd
import itertools
from fractions import Fraction

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
    return tuple(uint8tuple)#"({},{},{})".format(*uint8tuple)

def css_to_html(css):
    return f"<text style=background-color:{css}>&nbsp;&nbsp;&nbsp;&nbsp;</text>"

def n_colors(n=33):
    hues = infinite_hues()
    hsvs = itertools.chain.from_iterable(hue_to_hsvs(hue) for hue in hues)
    rgbs = (colorsys.hsv_to_rgb(*hsv) for hsv in hsvs)
    csss = (rgb_to_css(rgb) for rgb in rgbs)
    to_ret = list(itertools.islice(csss, n))
    return to_ret #[(float(c) for c in it) for it in to_ret]

class PWLinTraj:
    def __init__(self, path, duration):
        self.path = path
        self.duration = duration
        self.num_waypoints = len(self.path)

    def value(self, time):
        prog_frac = np.clip(time/self.duration, a_min = 0, a_max = 0.99999)*(self.num_waypoints-1)
        prog_int = int(prog_frac)
        prog_part = prog_frac-prog_int
        wp1 = self.path[prog_int]
        wp2 = self.path[prog_int+1]
        return wp1 + prog_part*(wp2-wp1)

    def end_time(self,):
        return self.duration

def animate(traj, publisher, steps, runtime):
    #loop
    idx = 0
    going_fwd = True
    time_points = np.linspace(0, traj.end_time(), steps) 

    for _ in range(runtime):
        #print(idx)
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
    #im using snopt, sue me
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

def plot_3d_poly_marchingcubes(region, resolution, vis, name, mat = None, verbose = False):
    
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

def plot_3d_poly(region, vis, name, mat = None, verbose = False):
    # First, prune region
    region = region.ReduceInequalities()

    region = region.ReduceInequalities()

    def project_and_triangulate(pts):
        n = np.cross(pts[0,:]-pts[1,:],pts[0,:]-pts[2,:])
        n = n / np.linalg.norm(n)
        if not np.abs(n[2]) < 1e-3:  # normal vector not in the XY-plane
            pts_prj = pts[:,:2]
        else:
            if np.abs(n[0]) < 1e-3:  # normal vector not in the Y-direction
                pts_prj = pts[:,[0,2]]
            else:
                pts_prj = pts[:,1:3]
        tri = Delaunay(pts_prj)
        return tri.simplices

    # Find feasible point in region
    # prog = MathematicalProgram()
    # x = prog.NewContinuousVariables(3)
    # prog.AddConstraint(le(region.A()@x, region.b()))
    # result = Solve(prog)
    # if result.is_success():
    #    x0 = result.GetSolution()
    # else:
    #    print("Solve failed. No feasible point found in region.")
    x0 = region.MaximumVolumeInscribedEllipsoid().center()
    A = region.A()
    b = region.b() - A@x0

    # Ax <= b must be in form [b -A]
    b_minusA = np.concatenate((b.reshape(-1,1),-A),axis=1)

    matrix = cdd.Matrix(b_minusA)
    matrix.rep_type = cdd.RepType.INEQUALITY
    poly = cdd.Polyhedron(matrix)
    # Get vrep
    gen = poly.get_generators()

    # [t vertices]
    t_v = np.array(gen)
    vertices = t_v[:,1:] + x0  # vertices need to be moved back

    # only keep nonempty facets
    facets_with_duplicates = [np.array(list(facet)) for facet in poly.get_input_incidence() if len(list(facet))>2]

    # if facet is subset of any other facet, remove
    remove_idxs = []
    for idx, facet in enumerate(facets_with_duplicates):
        for other_idx, other_facet in enumerate(facets_with_duplicates):
            if (not (idx == other_idx)) and set(facet).issubset(set(other_facet)):
                remove_idxs.append(idx)
    if remove_idxs:  # if list not empty 
        facets = list(np.delete(np.array(facets_with_duplicates), np.array(remove_idxs)))
    else:
        facets = facets_with_duplicates

    mesh_vertices = []
    mesh_triangles = []
    
    count = 0
    for idx, facet in enumerate(facets):
        tri = project_and_triangulate(vertices[facet])
        verts = vertices[facet]
        idxs = tri+count
    #    print('it:', idx)
    #   print(len(verts))
    #    print(len(idxs))
        mesh_vertices.append(verts)
        mesh_triangles.append(idxs)

        count += vertices[facet].shape[0]

    # print('end')
    mesh_vertices = np.concatenate(mesh_vertices, 0)
    mesh_triangles = np.concatenate(mesh_triangles, 0)

    if mat is None:
        mat = meshcat.geometry.MeshLambertMaterial(color=0x000000 , wireframe=False)
        mat.opacity = 0.3
    
    vis[name].set_object(
                meshcat.geometry.TriangularMeshGeometry(mesh_vertices, mesh_triangles),
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


def plot_regions(vis, regions, ellipses = None, region_suffix='', opacity = 0.5):
    colors = n_colors(len(regions))
    for i, region in enumerate(regions):
        c = colors[i]
        mat = meshcat.geometry.MeshLambertMaterial(color=rgb_to_hex(c), wireframe=False)
        mat.opacity = opacity
        # plot_3d_poly(region=region,
        #                   vis=vis['iris']['regions'+region_suffix],
        #                   name=str(i),
        #                   mat=mat)
        plot_3d_poly_marchingcubes(region=region,
                            resolution=30,
                            vis=vis['iris']['regions'+region_suffix],
                            name=str(i),
                            mat=mat)
        if ellipses is not None:
            C = ellipses[i].A()  # [:, (0,2,1)]
            d = ellipses[i].center()  # [[0,2,1]]
            radii, R = np.linalg.eig(C.T @ C)
            R[:, 0] = R[:, 0] * np.linalg.det(R)
            Rot = RotationMatrix(R)

            transf = RigidTransform(Rot, d)
            mat = meshcat.geometry.MeshLambertMaterial(color=rgb_to_hex(c), wireframe=True)
            mat.opacity = 0.15
            vis['iris']['ellipses'+region_suffix][str(i)].set_object(
                meshcat.geometry.Ellipsoid(np.divide(1, np.sqrt(radii))),
                mat)

            vis['iris']['ellipses'+region_suffix][str(i)].set_transform(transf.GetAsMatrix4())

    return vis



