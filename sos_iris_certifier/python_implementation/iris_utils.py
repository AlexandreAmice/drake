from pydrake.all import (
    HPolyhedron, VPolytope, Sphere, Ellipsoid,
    Iris, IrisOptions, MakeIrisObstacles, Variable
)
import pydrake.symbolic as sym
import numpy as np
from scipy.spatial import ConvexHull


def MakeFromHPolyhedronSceneGraph(query, geom, expressed_in=None):
    shape = query.inspector().GetShape(geom)
    if isinstance(shape, (Sphere, Ellipsoid)):
        raise ValueError(f"Sphere or Ellipsoid not Supported")
    return HPolyhedron(query, geom, expressed_in)
def MakeFromVPolytopeSceneGraph(query, geom, expressed_in=None):
    shape = query.inspector().GetShape(geom)
    if isinstance(shape, (Sphere, Ellipsoid)):
        raise ValueError(f"Sphere or Ellipsoid not Supported")
    return VPolytope(query, geom, expressed_in)

def _orth(M):
    # Find basis of M and its orthogonal subspace
    _, D, V = np.linalg.svd(M)
    return V[D >= 1e-9], V[D < 1e-9]

def extract_mono_verts(mono_list, indets):
    verts = []
    for m in mono_list:
        degs = [None] * len(indets)
        for ind, var in enumerate(indets):
            degs[ind] = m.degree(var)
        verts.append(degs)
    return verts

def NewContinuousVariablesWithoutProg(rows: int, cols: int, name: str):
    var_mat = np.array([[sym.Variable(f'{name}({r}, {c})') for c in range(cols)] for r in range(rows)])
    vars = var_mat.flatten()
    return var_mat, vars

def sparsest_sos_basis_poly(poly):
    """
    given a polynomial compute the monomial elements which lie in 1/2 Newt(p) which is the sparsest basis that can
    is necessary and sufficient to represent poly as SOS
    :param poly:
    :return:
    """
    # monomials in the polynomial
    monos = list(poly.monomial_to_coefficient_map().keys())
    indets = list(poly.indeterminates())
    return sparsest_sos_basis_by_given_basis(monos, indets)

def sparsest_sos_basis_by_given_basis(monos, indets):
    # adapted from https://github.com/yuanchenyang/SumOfSquares.py/blob/e20f8f59b0d78b50e882d353709a213fad739041/SumOfSquares/basis.py
    poly_verts = extract_mono_verts(monos, indets)
    poly_verts_div2 = np.array(poly_verts) / 2

    max_total_degree = np.max([m.total_degree() for m in monos])

    full_basis = sym.MonomialBasis(sym.Variables(indets), int(np.ceil(max_total_degree/ 2)))

    polytope_mean = np.mean(poly_verts_div2)
    U, U_ = _orth(poly_verts_div2 - polytope_mean)  # U orthogonal to U_
    proj, proj_ = lambda m: U.dot(m - polytope_mean), lambda m: U_.dot(m - polytope_mean)
    hull = ConvexHull(np.apply_along_axis(proj, 1, poly_verts_div2))

    def in_hull(pt):  # Point lies in affine subspace and convex hull
        return np.linalg.norm(proj_(pt)) < 1e-9 and \
               sum(hull.equations.dot(np.append(proj(pt), 1)) > 1e-9) == 0

    degs = list(filter(in_hull, extract_mono_verts(full_basis, indets)))
    basis = []
    for d in degs:
        m = {var: d[i] for i, var in enumerate(indets)}
        basis.append(sym.Monomial(m))
    return basis
