#pragma once

#include <unordered_map>
#include <utility>
#include <vector>

#include "drake/solvers/mathematical_program.h"
#include "drake/solvers/program_attribute.h"

namespace drake {
namespace solvers {

/** Given a convex program (specficially an LP, QP, SOCP, or SDP), construct an
 * equivalent program in standard primal conic form:
 * min        〈 c, x 〉+ d
 * subject to  Ax + b ∈ K
 * Where K is the product of cones.
 * @param prog
 * @return
 */
// std::unique_ptr<MathematicalProgram> ParseToConicStandardForm(
//    const MathematicalProgram& prog);

/**
 * Given a convex program (specficially an LP, QP, SOCP, or SDP), parse it into
 * an equivalent program in standard primal conic form:
 * min        〈 c, x 〉
 * subject to  Ax + b ∈ K
 * Where K is the product of the cones:
 * 1) Zero cone {x | x = 0 }
 * 2) Positive orthant {x | x ≥ 0 }
 * 3) Second-order cone {(t, x) | |x|₂ ≤ t }
 * 4) Positive semidefinite cone { X | min(eig(X)) ≥ 0, X = Xᵀ }
 *
 * [out] c a sparse vector encoding the cost.
 * [out] A is the sparse matrix encoding the constraint.
 * [out] b is the vector encoding the constraint
 * [out] attributes_to_start_end_pairs[type][i] is a pair of indices [start,
 * end) encoding that (Ax+b)[start:end, :] is in the cone corresponding to the
 * attribute type.
 *
 * Note that if (Ax+b)[start:end, :] is in the psd cone, then it is a vector
 * corresponding to the lower-triangular part of a PSD matrices with the
 * off-diagonal entires scaled by √2 to preserve inner products. For example
 * if (Ax+b)[start:end, :] = [y0, y1, y2, y3, y4, y5], then
 * Y = [y0,    y1/√2,  y2/√2]
 *     [y1/√2,   y3,  y4/√2]
 *     [y2/√2, y4/√2, y5]]
 *
 * Currently, I only support programs with linear costs.
 */
void ParseToConicStandardForm(
    const MathematicalProgram& prog, Eigen::SparseVector<double>* c, double* d,
    Eigen::SparseMatrix<double>* A, Eigen::SparseVector<double>* b,
    std::unordered_map<ProgramAttribute, std::vector<std::pair<int, int>>>*
        attributes_to_start_end_pairs);

}  // namespace solvers
}  // namespace drake
