import numpy as np

from pydrake.geometry.optimization import (
    GraphOfConvexSets,
    HPolyhedron,
)
from pydrake.math import BsplineBasis, BsplineBasis_, KnotVectorType
from pydrake.solvers.mathematicalprogram import (
    Binding,
    Constraint,
    Cost,
    L2NormCost,
    LinearConstraint,
)
from pydrake.symbolic import (
    DecomposeLinearExpressions,
    Expression,
    ExtractVariablesFromExpression,
    intersect,
    MakeMatrixContinuousVariable,
    MakeVectorContinuousVariable,
    Variables,
)
from pydrake.trajectories import BsplineTrajectory, BsplineTrajectory_

"""
Version of Andres bspline code
"""
def solveBsplineTrajectory(source, target, regions, rounding=False, max_velocity=None, order=6,
                           extra_control_points_per_region=0, max_repetitions=1):
    # Constructor
    assert len(source) == regions[-1].ambient_dimension()
    assert len(target) == regions[-1].ambient_dimension()
    for r in regions:
        assert r.ambient_dimension() == regions[-1].ambient_dimension()
    if max_velocity is not None:
        assert len(max_velocity) == len(source)
    else:
        max_velocity = np.inf * np.ones(len(source))
    regions.insert(0, HPolyhedron.MakeBox(source, source))
    regions.append(HPolyhedron.MakeBox(target, target))
    
    # Start building SPP
    spp = GraphOfConvexSets()
    
    # Add vertices to graph
    control_points_per_region = order - 1 + extra_control_points_per_region
    duration_scaling = HPolyhedron.MakeBox([0], [100])
    replicated_regions = []
    vertices = []
    for r in regions:
        full_space_region = r.CartesianPower(control_points_per_region).CartesianProduct(duration_scaling)
        replicated_regions.append(full_space_region)
        vertices.append(spp.AddVertex(full_space_region))

    # Formulate edge cost
    u_control = MakeMatrixContinuousVariable(len(source), control_points_per_region, "xu")
    v_control = MakeMatrixContinuousVariable(len(source), control_points_per_region, "xv")
    u_duration = MakeVectorContinuousVariable(1, "Tu")
    v_duration = MakeVectorContinuousVariable(1, "Tv")
    edge_control = []
    for ii in range(u_control.shape[1]):
        edge_control.append(u_control[:, ii])
    for ii in range(v_control.shape[1]):
        edge_control.append(v_control[:, ii])
    q_trajectory = BsplineTrajectory_[Expression](
        BsplineBasis_[Expression](order, 2*control_points_per_region, KnotVectorType.kUniform,
                                  0.0, order - 1 + 2*extra_control_points_per_region),
        edge_control)
    
    costs = []
    for deriv in range(1, order):
        deriv_traj = q_trajectory.MakeDerivative(deriv)
        for point in deriv_traj.control_points():
            for ii in range(len(source)):
                cost_vars = ExtractVariablesFromExpression(point[ii, 0])[0]
                if (len(intersect(Variables(cost_vars), Variables(u_control.flatten()))) > 0):
                    costs.append(point[ii, 0])
    costs.append(10 * u_duration[0])
    costs.append(10 * v_duration[0])
#     print(costs)
    
    edge_vars = np.concatenate((u_control.flatten("F"), u_duration, v_control.flatten("F"), v_duration))
    H = DecomposeLinearExpressions(costs, edge_vars)
#     print(H)
    H /= np.linalg.norm(H)
    edge_cost = L2NormCost(H, np.zeros(H.shape[0]))
    
    # Formulate velocity constraint
    v_trajectory = q_trajectory.MakeDerivative()
    vel_coeffs = []
    for point in v_trajectory.control_points():
        vel_con_vars = ExtractVariablesFromExpression(point[0, 0])[0]
        if (len(intersect(Variables(vel_con_vars), Variables(u_control.flatten()))) > 0):
            vel_coeffs.append(DecomposeLinearExpressions(point, edge_vars))
    vel_limit = DecomposeLinearExpressions(max_velocity * u_duration[0] / (order - 1 + 2*extra_control_points_per_region), edge_vars)
    
    # Add edges to graph and apply costs/constraints
    for ii in range(len(vertices)):
        for jj in range(ii + 1, len(vertices)):
            if vertices[ii].set().IntersectsWith(vertices[jj].set()):
                edge = spp.AddEdge(vertices[ii], vertices[jj], f"({ii}, {jj})")
                reverse_edge = spp.AddEdge(vertices[jj], vertices[ii], f"({jj}, {ii})")
                
                edge.AddCost(Binding[Cost](edge_cost, np.append(edge.xu(), edge.xv())))
                reverse_edge.AddCost(Binding[Cost](edge_cost, np.append(reverse_edge.xu(), reverse_edge.xv())))
                
                # Constraint first order - 1 points in xv to be in xu
                for col in range(order - 1):
                    edge.AddConstraint(Binding[Constraint](
                            LinearConstraint(regions[ii].A(), -np.inf*np.ones(len(regions[ii].b())),
                                             regions[ii].b()),
                            edge.xv()[len(source) * col:len(source) * (col+1)]))
                    reverse_edge.AddConstraint(Binding[Constraint](
                            LinearConstraint(regions[jj].A(), -np.inf*np.ones(len(regions[jj].b())), 
                                             regions[jj].b()),
                            reverse_edge.xv()[len(source) * col:len(source) * (col+1)]))
                     
                # Constraint duration scaling to match across verticies
                edge.AddConstraint(edge.xu()[-1] == edge.xv()[-1])
                reverse_edge.AddConstraint(reverse_edge.xu()[-1] == reverse_edge.xv()[-1])
                
                # Add velocity constraints
                if np.isfinite(max_velocity).all():
                    for coeffs in vel_coeffs:
                        edge.AddConstraint(Binding[Constraint](
                            LinearConstraint(coeffs - vel_limit, -np.inf*np.ones(len(source)),
                                             np.zeros(len(source))),
                            np.append(edge.xu(), edge.xv())))
                        edge.AddConstraint(Binding[Constraint](
                            LinearConstraint(-coeffs - vel_limit, -np.inf*np.ones(len(source)),
                                             np.zeros(len(source))),
                            np.append(edge.xu(), edge.xv())))
                        reverse_edge.AddConstraint(Binding[Constraint](
                            LinearConstraint(coeffs - vel_limit, -np.inf*np.ones(len(source)),
                                             np.zeros(len(source))),
                            np.append(reverse_edge.xu(), reverse_edge.xv())))
                        reverse_edge.AddConstraint(Binding[Constraint](
                            LinearConstraint(-coeffs - vel_limit, -np.inf*np.ones(len(source)),
                                             np.zeros(len(source))),
                            np.append(reverse_edge.xu(), reverse_edge.xv())))
    
    result = spp.SolveShortestPath(vertices[0], vertices[-1], rounding)
    print(f"Success: {result.get_solution_result()} Cost: {result.get_optimal_cost()}")
    
    for e in spp.Edges():
        print(e.name(), ":", result.GetSolution(e.phi()))
        
    if not result.is_success():
        return None, result, spp, None
    
    # Extract path with a tree walk
    active_edges = []
    max_phi = 0
    max_edge = None
    for edge in spp.Edges():
        phi = result.GetSolution(edge.phi())
        if edge.u() == vertices[0] and phi > max_phi:
            max_phi = phi
            max_edge = edge
    active_edges.append(max_edge)
    print("Added", max_edge.name(), "to path.")
    
    while active_edges[-1].v() != vertices[-1]:
        max_phi = 0
        max_edge = None
        for edge in spp.Edges():
            phi = result.GetSolution(edge.phi())
            if edge.u() == active_edges[-1].v() and phi > max_phi:
                max_phi = phi
                max_edge = edge
        active_edges.append(max_edge)
        print("Added", max_edge.name(), "to path.")
        
    # Solve with hard edge choices
    if rounding:
        for edge in spp.Edges():
            if edge in active_edges:
                edge.AddPhiConstraint(True)
            else:
                edge.AddPhiConstraint(False)
        hard_result = spp.SolveShortestPath(vertices[0], vertices[-1], rounding)
    else:
        hard_result = result
        
    # Extract trajectory control points
    control_points = []
    vertex_points = np.reshape(hard_result.GetSolution(active_edges[0].xu())[:-1],
                               (len(source), control_points_per_region), "F")
    for ii in range(vertex_points.shape[1]):
        control_points.append(vertex_points[:, ii])
    for edge in active_edges:
        vertex_points = np.reshape(hard_result.GetSolution(edge.xv())[:-1],
                                   (len(source), control_points_per_region), "F")
        for ii in range(vertex_points.shape[1]):
            control_points.append(vertex_points[:, ii])
            
#     for edge in active_edges:
#         print("Decision variables for edge", edge.name())
#         print(np.reshape(hard_result.GetSolution(edge.xu())[:-1], (len(source), control_points_per_region), "F"))
#         print(np.reshape(hard_result.GetSolution(edge.xv())[:-1], (len(source), control_points_per_region), "F"))
#         print(hard_result.GetSolution(edge.xu())[-1], hard_result.GetSolution(edge.xv())[-1])
#     print (control_points)
    
    nominal_duration = (len(control_points) - (order - 1.0)) / (order - 1.0 + 2*extra_control_points_per_region);
    duration_scaling = hard_result.GetSolution(active_edges[-1].xu()[-1]);

    traj = BsplineTrajectory(BsplineBasis(order, len(control_points), KnotVectorType.kUniform,
                                          0, nominal_duration * duration_scaling),
                             control_points);
    
    return traj, result, spp, hard_result