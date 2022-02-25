from pydrake.all import (
    ConvexSet, HPolyhedron, Hyperellipsoid,
    MathematicalProgram, Solve, le, IpoptSolver,
    Role, Sphere, VPolytope,
    Iris, IrisOptions, MakeIrisObstacles, Variable,
    BsplineTrajectoryThroughUnionOfHPolyhedra,
    eq, SnoptSolver,
    Sphere, Ellipsoid, GeometrySet,
    RigidBody_, AutoDiffXd, initializeAutoDiff, InverseKinematics,
    RationalForwardKinematics, FindBodyInTheMiddleOfChain
)
import sys
import os
import time
import numpy as np
from functools import partial
import itertools
import pydrake
import meshcat

from pydrake.all import BsplineTrajectoryThroughUnionOfHPolyhedra, IrisInConfigurationSpace, IrisOptions
from pydrake.common import FindResourceOrThrow
from pydrake.geometry import SceneGraph
from pydrake.math import RigidTransform, RollPitchYaw
from pydrake.multibody.optimization import CalcGridPointsOptions, Toppra
from pydrake.multibody.parsing import LoadModelDirectives, Parser, ProcessModelDirectives
from pydrake.multibody.plant import MultibodyPlant, AddMultibodyPlantSceneGraph
from pydrake.multibody.tree import RevoluteJoint
from pydrake.solvers.mathematicalprogram import MathematicalProgram, Solve
from pydrake.solvers.mosek import MosekSolver
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.primitives import TrajectorySource
from pydrake.trajectories import PiecewisePolynomial
from pydrake.all import Variable, Expression, RotationMatrix
from pydrake.all import MultibodyPositionToGeometryPose, ConnectMeshcatVisualizer, Role, Sphere
from pydrake.all import (
    ConvexSet, HPolyhedron, Hyperellipsoid,
    MathematicalProgram, Solve, le, IpoptSolver,
    Role, Sphere, VPolytope,
    Iris, IrisOptions, MakeIrisObstacles, Variable
)
from pydrake.all import (
    eq, SnoptSolver,
    Sphere, Ellipsoid, GeometrySet,
    RigidBody_, AutoDiffXd, initializeAutoDiff, InverseKinematics
)

import pydrake.symbolic as sym
import symbolic_parsing_helpers as symHelpers
from pydrake.all import RationalForwardKinematics, FindBodyInTheMiddleOfChain
from pydrake.all import GenerateMonomialBasisOrderAllUpToOneExceptOneUpToTwo, GenerateMonomialBasisWithOrderUpToOne

import scipy
import visualizations_utils


def set_up_iris_t_space(plant, scene_graph, context, settings = None):
    

    #hardcoded
    dReal_polytope_tol = 1e-10
    starting_vol_eps = 1e-3


    forward_kin = RationalForwardKinematics(plant)
    query = scene_graph.get_query_output_port().Eval(scene_graph.GetMyContextFromRoot(context))
    q_star = np.zeros(forward_kin.t().shape[0])

    def convert_RationalForwardPoseList_to_TransformExpressionList(pose_list):
        ret = []
        for p in pose_list:
            ret.append(p.asRigidTransformExpr())
        return ret

    def MakeFromSceneGraph(query, geom, expressed_in=None):
        shape = query.inspector().GetShape(geom)
        if isinstance(shape, (Sphere, Ellipsoid)):
            return Hyperellipsoid(query, geom, expressed_in)
        return HPolyhedron(query, geom, expressed_in)

    def GrowthVolumeRational(E, X_WA, X_WB, setA, setB, A, b, guess=None):
        snopt = SnoptSolver()
        prog = MathematicalProgram()
        t = forward_kin.t()
        prog.AddDecisionVariables(t)
        
        if guess is not None:
            prog.SetInitialGuess(t, guess)

        prog.AddLinearConstraint(A, b-np.inf, b, t)
        p_AA = prog.NewContinuousVariables(3, "p_AA")
        p_BB = prog.NewContinuousVariables(3, "p_BB")
        setA.AddPointInSetConstraints(prog, p_AA)
        setB.AddPointInSetConstraints(prog, p_BB)
        prog.AddQuadraticErrorCost(E.A().T @ E.A(), E.center(), t)

        p_WA = X_WA.multiply(p_AA+0)

        p_WB = X_WB.multiply(p_BB+0)

        prog.AddConstraint(eq(p_WA, p_WB))
        result = snopt.Solve(prog)
        return result.is_success(), result.get_optimal_cost(), result.GetSolution(t)

    def TangentPlane(self, point):
        a = 2 * self.A().T @ self.A() @ (point - self.center())
        a = a / np.linalg.norm(a)
        b = a.dot(point)
        return a, b

    Hyperellipsoid.TangentPlane = TangentPlane

    def iris_rational_space(query, point, require_containment_points=[], termination_threshold=2e-2, iteration_limit=100):
        dim = plant.num_positions()
        lb = plant.GetPositionLowerLimits()
        rational_lb = forward_kin.ComputeTValue(lb, q_star)
        ub = plant.GetPositionUpperLimits()
        rational_ub = forward_kin.ComputeTValue(ub, q_star)
        volume_of_unit_sphere = 4.0*np.pi/3.0
        
        E = Hyperellipsoid(np.eye(3)/starting_vol_eps, point)

        best_volume = starting_vol_eps**dim * volume_of_unit_sphere
        

        max_faces = 10
        
        link_poses_by_body_index_rat_pose = forward_kin.CalcLinkPoses(q_star, 
                                                            plant.world_body().index())
        X_WA_list = convert_RationalForwardPoseList_to_TransformExpressionList(link_poses_by_body_index_rat_pose)
        X_WB_list = convert_RationalForwardPoseList_to_TransformExpressionList(link_poses_by_body_index_rat_pose)

        inspector = query.inspector()
        pairs = inspector.GetCollisionCandidates()

        P = HPolyhedron.MakeBox(rational_lb, rational_ub)
        A = np.vstack((P.A(), np.zeros((max_faces*len(pairs),3))))  # allow up to 10 faces per pair.
        b = np.concatenate((P.b(), np.zeros(max_faces*len(pairs))))

        # geom_ids = inspector.GetGeometryIds(GeometrySet(inspector.GetAllGeometryIds()), Role.kProximity)
        pair_set = set()
        for p in pairs:
            pair_set.add(p[0])
            pair_set.add(p[1])
        geom_ids = inspector.GetGeometryIds(GeometrySet(list(pair_set)))

        sets = {geom:MakeFromSceneGraph(query, geom, inspector.GetFrameId(geom)) for geom in geom_ids}
        body_indexes_by_geom_id = {geom:
                                plant.GetBodyFromFrameId(inspector.GetFrameId(geom)).index() for geom in geom_ids} 
        
        #Turn onto true to certify regions using SOS at each iteration.
        certify = False
        # refine polytopes if not certified collision free
        refine = False and certify
            
        iteration = 0
        num_faces = 2*len(lb)
        while True:
            ## Find separating hyperplanes

            for geomA, geomB in pairs:
                #print(f"geomA={inspector.GetName(geomA)}, geomB={inspector.GetName(geomB)}")
                # Run snopt at the beginning
                while True:
                    X_WA = X_WA_list[int(body_indexes_by_geom_id[geomA])]
                    X_WB = X_WB_list[int(body_indexes_by_geom_id[geomB])]
                    success, growth, qstar = GrowthVolumeRational(E,
                        X_WA, X_WB,
                        sets[geomA], sets[geomB], 
                        A[:num_faces,:], b[:num_faces] - dReal_polytope_tol, 
                        point)
                    if success:
                        #print(f"snopt_example={qstar}, growth = {growth}")
                        # Add a face to the polytope
                        A[num_faces,:], b[num_faces] = E.TangentPlane(qstar)
                        num_faces += 1
                        if num_faces > max_faces:
                            break
                        #     A = np.vstack((A, np.zeros((max_faces*len(pairs),3))))  # allow up to 10 faces per pair.
                        #     b = np.concatenate((b, np.zeros(max_faces*len(pairs))))
                    else:
                        break

                if certify:
                    pass
                

            if any([np.any(A[:num_faces,:] @ p > b[:num_faces]) for p in require_containment_points]):
                print("terminating because a required containment point would have not been contained")
                break

            P = HPolyhedron(A[:num_faces,:],b[:num_faces])

            E = P.MaximumVolumeInscribedEllipsoid()
            print(iteration)

            iteration += 1
            if iteration >= iteration_limit:
                break

            volume = volume_of_unit_sphere / np.linalg.det(E.A())
            if volume - best_volume <= termination_threshold:
                break
            best_volume = volume

        return P
    return iris_rational_space, query, forward_kin

def MakeFromHPolyhedronOrEllipseSceneGraph(query, geom, expressed_in=None):
    shape = query.inspector().GetShape(geom)
    if isinstance(shape, (Sphere, Ellipsoid)):
        return Hyperellipsoid(query, geom, expressed_in)
    return HPolyhedron(query, geom, expressed_in)
def MakeFromVPolytopeOrEllipseSceneGraph(query, geom, expressed_in=None):
    shape = query.inspector().GetShape(geom)
    if isinstance(shape, (Sphere, Ellipsoid)):
        return Hyperellipsoid(query, geom, expressed_in)
    return VPolytope(query, geom, expressed_in)

def EvaluatePlanePair(plane_pair, eval_dict):
    a_res = []
    for ai in plane_pair[0]:
        a_res.append(ai.Evaluate(eval_dict))
    return (np.array(a_res), plane_pair[1].Evaluate(eval_dict))

def uniform_shrink_iris_region(region, var_epsilon):
    return HPolyhedron(region.A(), region.b()-var_epsilon*np.ones_like(region.b()))

def uniform_shrink_iris_region_list(region_list, var_epsilon):
    return [uniform_shrink_iris_region(r, var_epsilon) for r in region_list]

class RegionCertifier:
    def __init__(self, plant, scene_graph, context):
        self.query = scene_graph.get_query_output_port().Eval(scene_graph.GetMyContextFromRoot(context))

        self.inspector = self.query.inspector()
        self.pairs = self.inspector.GetCollisionCandidates()

        self.geom_ids = self.inspector.GetGeometryIds(GeometrySet(self.inspector.GetAllGeometryIds()), Role.kProximity)
        self.HPolyhedronSets = {geom: MakeFromHPolyhedronOrEllipseSceneGraph(self.query, geom, self.inspector.GetFrameId(geom)) for
                           geom in self.geom_ids}
        self.VPolyhedronSets = {geom: MakeFromVPolytopeOrEllipseSceneGraph(self.query, geom, self.inspector.GetFrameId(geom)) for geom
                           in self.geom_ids}

        self.body_indexes_by_geom_id = {geom:
                                       plant.GetBodyFromFrameId(self.inspector.GetFrameId(geom)).index() for geom in
                                   self.geom_ids}
        self.forward_kin = RationalForwardKinematics(plant)
        self.convSolver = MosekSolver()
        self.t_kin = self.forward_kin.t()

        self.link_poses_by_body_index_multilinear_pose = self.forward_kin.CalcLinkPosesAsMultilinearPolynomials(np.zeros_like(self.t_kin),
                                                                                                      plant.world_body().index())
        self.X_WA_multilinear_list = [(r.rotation().copy(), r.translation().copy()) for r in
                                 self.link_poses_by_body_index_multilinear_pose]

        def convert_RationalForwardPoseList_to_TransformExpressionList(pose_list):
            ret = []
            for p in pose_list:
                ret.append(p.asRigidTransformExpr())
            return ret

        X_WA_list = convert_RationalForwardPoseList_to_TransformExpressionList(link_poses_by_body_index_rat_pose)

        # t_space_vertex_position_by_geom_id = {}
        # for geom in geom_ids:
        #     VPoly = VPolyhedronSets[geom]
        #     num_verts = VPoly.vertices().shape[1]
        #     X_WA = X_WA_list[int(body_indexes_by_geom_id[geom])]
        #     R_WA = X_WA.rotation().matrix()
        #     p_WA = X_WA.translation()
        #     vert_pos = R_WA @ (VPoly.vertices()) + np.repeat(p_WA[:, np.newaxis], num_verts, 1)
        #     t_space_vertex_position_by_geom_id[geom] = vert_pos

    def construct_separating_hyperplane_of_order(self, prog, t, order=2, plane_name=''):
        if plane_name != '':
            plane_name = '_' + plane_name
        t_basis = sym.MonomialBasis(t, order)
        t_basis = np.array([sym.Polynomial(v) for v in t_basis])
        a_A_coeffs = prog.NewContinuousVariables(3, t_basis.shape[0], 'a_A' + plane_name)
        a_poly = a_A_coeffs @ t_basis

        b_A_coeffs = prog.NewContinuousVariables(1, t_basis.shape[0], 'b_A' + plane_name)
        b_poly = b_A_coeffs @ t_basis

        for i, p in enumerate(a_poly):
            a_poly[i].SetIndeterminates(sym.Variables(t))
        for i, p in enumerate(b_poly):
            b_poly[i].SetIndeterminates(sym.Variables(t))
        dec_vars = [*a_A_coeffs.flatten().tolist(), *b_A_coeffs.flatten().tolist()]
        ntmp = a_A_coeffs.flatten().shape[0]
        Qtmp, btmp = np.eye(ntmp), np.zeros(ntmp)
        prog.AddQuadraticCost(Qtmp, btmp, a_A_coeffs.flatten())
        return prog, a_poly, b_poly.item(), dec_vars


    def putinarPsatConstraint(self, prog, p, t,
                              poly_to_cert, lagrange_mult_degree=2, var_epsilon=None):
        A = poly_to_cert.A()
        b = poly_to_cert.b()
        n = b.shape[0]
        if var_epsilon is None:
            var_epsilon = np.zeros(n)
        #     lagrange_poly, Q = prog.NewSosPolynomial(sym.Variables(t), lagrange_mult_degree)
        lagrange_poly, Q = prog.NewSosPolynomial(GenerateMonomialBasisOrderAllUpToOneExceptOneUpToTwo(
            sym.Variables(t)))
        lagrange_poly.SetIndeterminates(sym.Variables(t))
        prog.AddSosConstraint(lagrange_poly)
        constraint_poly = lagrange_poly
        for i in range(n):
            #         lagrange_poly, Q = prog.NewSosPolynomial(sym.Variables(t), lagrange_mult_degree)
            lagrange_poly, Q = prog.NewSosPolynomial(
                GenerateMonomialBasisOrderAllUpToOneExceptOneUpToTwo(sym.Variables(t)))
            lagrange_poly.SetIndeterminates(sym.Variables(t))
            prog.AddSosConstraint(lagrange_poly)
            constraint_poly += lagrange_poly * sym.Polynomial(b[i] - var_epsilon[i] - A[i, :] @ t)
        constraint_poly.SetIndeterminates(sym.Variables(t))
        tol = 1e-5
        prog.AddEqualityConstraintBetweenPolynomials(constraint_poly, p - tol)
        return prog, (constraint_poly, p - tol)


    def makeBodyHyperplaneSidePolynomials(self, prog, a_plane_poly, b_plane_poly,
                                          VPoly, R_WA, p_WA, t, poly_to_cert, leq_or_geq,
                                          lagrange_mult_degree=2, var_epsilon=None, base_point_as_multilinear_poly=None):
        num_verts = VPoly.vertices().shape[1]

        vertex_pos = R_WA @ (VPoly.vertices()) + np.repeat(p_WA[:, np.newaxis], num_verts, 1)
        zero_poly = sym.Polynomial(0)

        dens = np.ndarray(shape=vertex_pos.shape, dtype=object)
        nums = np.ndarray(shape=vertex_pos.shape, dtype=object)
        for i, row in enumerate(vertex_pos):

            for j, v in enumerate(row):
                vertex_pos[i, j] = self.forward_kin.ConvertMultilinearPolynomialToRationalFunction(v)
                dens[i, j] = vertex_pos[i, j].denominator()
                nums[i, j] = vertex_pos[i, j].numerator()

        unique_dens_c = [[sym.Polynomial(1)] for _ in range(dens.shape[1])]
        col_den = [sym.Polynomial(1) for _ in range(dens.shape[1])]
        for c in range(dens.shape[1]):
            for r in range(dens.shape[0]):
                is_unique = True
                for d in unique_dens_c[c]:
                    is_unique = False if dens[r, c].EqualTo(d) else True
                    if not is_unique:
                        break
                if is_unique:
                    unique_dens_c[c].append(dens[r, c])

            for d in unique_dens_c[c]:
                col_den[c] *= d

        for c in range(nums.shape[1]):
            for r in range(nums.shape[0]):
                for d in unique_dens_c[c]:
                    if not dens[r, c].EqualTo(d):
                        nums[r, c] *= d

        plane_polys = np.array([None for _ in range(vertex_pos.shape[1])])
        for i in range(vertex_pos.shape[1]):
            if leq_or_geq == 'leq':
                # a^Tx + b <= -1 -> -(a^Tx+b+1)>=0
                plane_polys[i] = -a_plane_poly.dot(nums[:, i]) - (b_plane_poly + 1) * (col_den[i])

            elif leq_or_geq == 'geq':
                # a^Tx+b >= 1 -> a^Tx+b -1 >= 0
                plane_polys[i] = a_plane_poly.dot(nums[:, i]) + (b_plane_poly - 1) * (col_den[i])
            else:
                raise ValueError("leq_or_geq arg must be leq or geq not {}".format(leq_or_geq))
            plane_polys[i].SetIndeterminates(sym.Variables(t))

        return plane_polys


    def add_pair_constraint(self, geomA, geomB, prog, poly_to_cert,
                            lagrange_mult_degree=2, var_epsilon=None):
        VPolyA, VPolyB = self.VPolyhedronSets[geomA], self.VPolyhedronSets[geomB]
        prog, a_plane_poly, b_plane_poly, dec_vars = self.construct_separating_hyperplane_of_order(prog, self.t_kin)

        R_WA, p_WA = self.X_WA_multilinear_list[int(self.body_indexes_by_geom_id[geomA])]
        R_WB, p_WB = self.X_WA_multilinear_list[int(self.body_indexes_by_geom_id[geomB])]

        base_point_poly = None  # R_WA@VPolyA.vertices().mean(axis = 1)-p_WA
        #     base_point = np.array([forward_kin.ConvertMultilinearPolynomialToRationalFunction(p) for p in base_point_poly])

        plane_polys_A = self.makeBodyHyperplaneSidePolynomials(prog, a_plane_poly, b_plane_poly,
                                                          VPolyA, R_WA, p_WA, self.t_kin,
                                                          poly_to_cert, 'leq', lagrange_mult_degree,
                                                          var_epsilon, base_point_poly)
        plane_polys_B = self.makeBodyHyperplaneSidePolynomials(prog, a_plane_poly, b_plane_poly,
                                                          VPolyB, R_WB, p_WB, self.t_kin,
                                                          poly_to_cert, 'geq', lagrange_mult_degree,
                                                          var_epsilon, base_point_poly)
        plane_polys = np.array(plane_polys_A.tolist() + plane_polys_B.tolist()).squeeze()

        s_prod_pairs = []

        for p in plane_polys:
            prog, (constraint_poly, p_with_tol) = self.putinarPsatConstraint(prog, p, self.t_kin,
                                                                        poly_to_cert,
                                                                        lagrange_mult_degree=lagrange_mult_degree,
                                                                        var_epsilon=var_epsilon)
            s_prod_pairs.append((constraint_poly, p_with_tol))
        return prog, plane_polys, (a_plane_poly, b_plane_poly), dec_vars, s_prod_pairs


    def construct_region_safety_problem(self, poly_to_cert, lagrange_mult_degree=2, var_epsilon=None, check_archimedean=True,
                                        plane_name=""):
        if check_archimedean:
            is_Archimedean, _ = self.certify_archimedean(poly_to_cert, self.t_kin)
            if not is_Archimedean:
                raise ValueError("region is not archimedean")
        prog = MathematicalProgram()
        prog.AddIndeterminates(self.t_kin)
        plane_pairs = {}
        plane_polys_list = []
        s_prod_pairs = {}
        dec_vars = []
        for i, (geomA, geomB) in enumerate(self.pairs):
            print("pair {}/{}".format(i + 1, len(self.pairs)))
            prog, plane_polys, (a_poly, b_poly), dec_vars0, s_prod_pairs0 = self.add_pair_constraint(geomA, geomB,
                                                                                                prog, poly_to_cert,
                                                                                                lagrange_mult_degree=lagrange_mult_degree,
                                                                                                var_epsilon=var_epsilon
                                                                                                )
            plane_polys_list.append(plane_polys)
            plane_pairs[(geomA, geomB)] = (a_poly, b_poly)
            s_prod_pairs[(geomA, geomB)] = s_prod_pairs0
            dec_vars += dec_vars0
        #         if i >= 0:
        #             break
        return prog, plane_pairs, plane_polys_list, np.array(dec_vars), s_prod_pairs


    def certify_region(self, poly_to_cert, lagrange_mult_degree=4, var_epsilon=None, check_archimedean=True, numeric_tol=1e-10):
        prog, plane_pairs, plane_polys_list, dec_vars, s_prod_pairs = self.construct_region_safety_problem(poly_to_cert,
                                                                                                      lagrange_mult_degree=lagrange_mult_degree,
                                                                                                      var_epsilon=var_epsilon,
                                                                                                      check_archimedean=check_archimedean)
        result = self.convSolver.Solve(prog)
        numerically_cert = result.is_success() and (np.linalg.norm(result.GetSolution(dec_vars)) >= numeric_tol)
        print(np.linalg.norm(result.GetSolution(dec_vars)))

        plane_polys = self.extract_planes(plane_pairs, result)
        s_prod_pairs = self.extract_s_prod_pairs(s_prod_pairs, result)
        return result, plane_polys, s_prod_pairs


    def certify_archimedean(self, poly_to_cert, t, lagrange_mult_degree=2):
        # need to fix
        convSolver = MosekSolver()
        prog = MathematicalProgram()
        prog.AddIndeterminates(t)

        t_bar = sym.Polynomial(t @ t)

        poly = t_bar
        A = poly_to_cert.A()
        b = poly_to_cert.b()
        n = b.shape[0]
        for i in range(n):
            lagrange_poly, Q = prog.NewSosPolynomial(sym.Variables(t), lagrange_mult_degree)
            lagrange_poly.SetIndeterminates(sym.Variables(t))
            prog.AddSosConstraint(lagrange_poly)
            poly += lagrange_poly * sym.Polynomial(b[i] - A[i, :] @ t)
        poly.SetIndeterminates(sym.Variables(t))
        prog.AddSosConstraint(poly)
        result = convSolver.Solve(prog)
        return result.is_success(), prog

    def extract_planes(self, plane_pairs, result):
        resulting_plane_pairs = {}

        for k, (a, b) in plane_pairs.items():
            a_list = []
            for ai in a:
                a_list.append(result.GetSolution(ai))
            resulting_plane_pairs[k] = (np.array(a_list), result.GetSolution(b))
        return resulting_plane_pairs


    def extract_s_prod_pairs(self, s_prod_pairs, result):
        evaled_pairs = {}
        for k, poly_list in s_prod_pairs.items():
            cur_list = []
            for (const, vert) in poly_list:
                constraint_poly_evaled = result.GetSolution(const)
                vert_poly_evaled = result.GetSolution(vert)
                cur_list.append((constraint_poly_evaled, vert_poly_evaled))
            evaled_pairs[k] = cur_list
        return evaled_pairs

class PlaneVisualizer():
    def __init__(self, t_kin, vis):
        self.vis = vis
        self.t_kin = t_kin
        self.x = np.linspace(-1, 1, 3)
        self.y = np.linspace(-1, 1, 3)
        self.verts = []

        for idxx in range(len(self.x)):
            for idxy in range(len(self.y)):
               self. verts.append(np.array([self.x[idxx], self.y[idxy]]))

        self.tri = scipy.spatial.Delaunay(self.verts)
        self.plane_triangles = self.tri.simplices
        self.plane_verts = self.tri.points[:, :]
        self.plane_verts = np.concatenate((self.plane_verts, 0 *self. plane_verts[:, 0].reshape(-1, 1)), axis=1)

    def transform(self, a, b, p1, p2, plane_verts, plane_triangles):
        alpha = (-b - a.T @ p1) / (a.T @ (p2 - p1))
        offset = alpha * (p2 - p1) + p1
        z = np.array([0, 0, 1])
        crossprod = np.cross(utils.normalize(a), z)
        if np.linalg.norm(crossprod) <= 1e-4:
            R = np.eye(3)
        else:
            ang = np.arcsin(np.linalg.norm(crossprod))
            axis = utils.normalize(crossprod)
            R = utils.get_rotation_matrix(axis, -ang)

        verts_tf = (R @ plane_verts.T).T + offset
        return verts_tf

    def transform_at_t(self, cur_t, a_poly, b_poly, p1_rat, p2_rat):
        eval_dict = dict(zip(sym.Variables(self.t_kin), cur_t))
        a, b = EvaluatePlanePair((a_poly, b_poly), eval_dict)
        #     print(f"{a}, {b}")
        p1 = np.array([p.Evaluate(eval_dict) for p in p1_rat])
        p2 = np.array([p.Evaluate(eval_dict) for p in p2_rat])
        return self.transform(a, b, p1, p2, self.plane_verts, self.plane_triangles), p1, p2

    def transform_plane_geom_id(self, geomA, geomB, planes_dict, cur_t):
        vA = self.t_space_vertex_position_by_geom_id[geomA][:, 0]
        vB = self.t_space_vertex_position_by_geom_id[geomB][:, 0]
        a_poly, b_poly = planes_dict[(geomA, geomB)]
        return self.transform_at_t(cur_t, a_poly, b_poly, vA, vB)

    def plot_plane_geom_id(self, geomA, geomB, planes_dict, cur_t):
        verts_tf, p1, p2 = self.transform_plane_geom_id(geomA, geomB, planes_dict, cur_t)

        mat = meshcat.geometry.MeshLambertMaterial(color=utils.rgb_to_hex((255, 0, 0)), wireframe=False)
        mat.opacity = 0.5
        self.vis["plane"][f"{geomA.get_value()}, {geomB.get_value()}"].set_object(
            meshcat.geometry.TriangularMeshGeometry(verts_tf, self.plane_triangles),
            mat)

        mat.opacity = 1.0
        utils.plot_point(loc=p1, radius=0.05, mat=mat, vis=self.vis["plane"][f"{geomA.get_value()}, {geomB.get_value()}"],
                         marker_id='p1')
        mat = meshcat.geometry.MeshLambertMaterial(color=utils.rgb_to_hex((255, 255, 0)), wireframe=False)
        mat.opacity = 1.0
        utils.plot_point(loc=p2, radius=0.05, mat=mat, vis=self.vis["plane"][f"{geomA.get_value()}, {geomB.get_value()}"],
                         marker_id='p2')

