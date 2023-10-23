#pragma once

#include <memory>
#include <optional>

#include "drake/solvers/mathematical_program.h"
#include "drake/solvers/solver_interface.h"

namespace drake {
namespace solvers {
namespace test {

// Tests a trivial semidefinite problem.
// min S(0, 0) + S(1, 1)
// s.t S(1, 0) = 1
//     S is p.s.d
// The analytical solution is
// S = [1 1]
//     [1 1]
// @return The tested program
std::unique_ptr<MathematicalProgram> TestTrivialSDP(
    const SolverInterface& solver, double tol, bool solver_is_available);

// Solves a semidefinite programming problem.
// Finds the common Lyapunov function for linear systems
// xdot = Ai*x
// The condition is
// min 0
// s.t P is positive definite
//     - (Ai'*P + P*Ai) is positive definite
// @return The tested program
std::unique_ptr<MathematicalProgram> FindCommonLyapunov(
    const SolverInterface& solver,
    const std::optional<SolverOptions>& solver_options, double tol,
    bool solver_is_available);

// Given some ellipsoids ℰᵢ : xᵀQᵢx + 2 bᵢᵀx ≤ 1, i = 1, 2, ..., n, finds an
// ellipsoid xᵀPx + 2cᵀx ≤ 1 as an outer approximation for the union of
// ellipsoids ℰᵢ.
//
// Using s-lemma, the ellipsoid xᵀPx + 2cᵀx ≤ 1 contains the ellipsoid ℰᵢ,
// if and only if there exists a scalar sᵢ ≥ 0 such that
//
// (1 - xᵀPx - cᵀx) - sᵢ(1 - xᵀQᵢx - bᵢᵀx) ≥ 0 ∀x.
//
// This is equivalent to requiring that the matrix
//
// ⎡sᵢQᵢ - P   sᵢbᵢ - c⎤
// ⎣sᵢbᵢᵀ - cᵀ   1 - sᵢ⎦
//
// is positive semidefinite.
//
// In order to find a tight outer approximation, we choose to maximize the
// trace of P. The optimization problem becomes
//
// min_{P, c, si} -trace(P)
// s.t ⎡sᵢQᵢ - P   sᵢbᵢ - c⎤ is p.s.d
//    ⎣sᵢbᵢᵀ - cᵀ   1 - sᵢ⎦
// P is p.s.d
// @return The tested program
std::unique_ptr<MathematicalProgram> FindOuterEllipsoid(
    const SolverInterface& solver,
    const std::optional<SolverOptions>& solver_options, double tol,
    bool solver_is_available);

// Solves an eigen value problem through a semidefinite programming.
// Minimize the maximum eigen value of a matrix that depends affinely on a
// variable x
// min  z
// s.t z * Identity - x1 * F1 - ... - xn * Fn is p.s.d
//     A * x <= b
//     C * x = d
// @return The tested program
std::unique_ptr<MathematicalProgram> SolveEigenvalueProblem(
    const SolverInterface& solver,
    const std::optional<SolverOptions>& solver_options, double tol,
    bool solver_is_available);

// Solves an SDP with a second order cone constraint. This example is taken
// from https://docs.mosek.com/10.0/capi/tutorial-sdo-shared.html
// @return The tested program
std::unique_ptr<MathematicalProgram> SolveSDPwithSecondOrderConeExample1(
    const SolverInterface& solver, double tol, bool solver_is_available);

// Solves an SDP with second order cone constraints. Notice that the variables
// appear in the second order cone constraints appear also in the positive
// semidefinite constraint.
// min X(0, 0) + X(1, 1) + x(0)
// s.t X(0, 0) + 2 * X(1, 1) + X(2, 2) + 3 * x(0) = 3
//     X(0, 0) >= sqrt((X(1, 1) + x(0))² + (X(1, 1) + X(2, 2))²)
//     X(1, 0) + X(2, 1) = 1
//     X is psd, x(0) >= 0
// @return The tested program
std::unique_ptr<MathematicalProgram> SolveSDPwithSecondOrderConeExample2(
    const SolverInterface& solver, double tol, bool solver_is_available);

// Solves an SDP with two PSD constraint, where each PSD constraint has
// duplicate entries and the two PSD matrix share a common variables.
// min 2 * x0 + x2
// s.t [x0 x1] is psd
//     [x1 x0]
//     [x0 x2] is psd
//     [x2 x0]
//     x1 == 1
// The optimal solution will be x = (1, 1, -1).
// @return The tested program
std::unique_ptr<MathematicalProgram> SolveSDPwithOverlappingVariables(
    const SolverInterface& solver, double tol, bool solver_is_available);

// Solves an SDP with quadratic cost and two PSD constraints, where each PSD
// constraint has duplicate entries and the two PSD matrix share a common
// variables.
// min x0² + 2*x0 + x2
// s.t ⎡x0 x1⎤ is psd
//     ⎣x1 x0⎦
//     ⎡x0 x2⎤ is psd
//     ⎣x2 x0⎦
//     x1 == 1
//
// The optimal solution will be x = (1, 1, -1).
// @return The tested program
std::unique_ptr<MathematicalProgram> SolveSDPwithQuadraticCosts(
    const SolverInterface& solver, double tol, bool solver_is_available);

// Tests a simple SDP with only PSD constraint and bounding box constraint.
// min x1
// s.t ⎡x0 x1⎤ is psd
//     ⎣x1 x2⎦
//     x0 <= 4
//     x2 <= 1
// @return The tested program
std::unique_ptr<MathematicalProgram> TestSDPDualSolution1(
    const SolverInterface& solver, double tol, bool solver_is_available);
}  // namespace test
}  // namespace solvers
}  // namespace drake
