#pragma once

#include <vector>

#include <Eigen/Dense>

namespace drake {

/// Returns true if and only if the two matrices are equal to within a certain
/// relative elementwise @p tolerance.  Special values (infinities, NaN, etc.)
/// do not compare as equal elements.
template <typename DerivedA, typename DerivedB>
bool is_approx_equal_reltol(const Eigen::MatrixBase<DerivedA>& m1,
                            const Eigen::MatrixBase<DerivedB>& m2,
                            double tolerance) {
  const Eigen::MatrixBase<DerivedA> abs_diff =
      (m1 - m2).template lpNorm<Eigen::Infinity>();
  const Eigen::MatrixBase<DerivedA> denom =
      m1.template cwiseAbs().template cwiseMin(m2.cwiseAbs());
  return ((m1.rows() == m2.rows()) && (m1.cols() == m2.cols()) &&
          (abs_diff.template cwiseProduct(denom.template cwiseInverse())
               .template lpNorm<Eigen::Infinity>() <= tolerance));
}

}  // namespace drake
