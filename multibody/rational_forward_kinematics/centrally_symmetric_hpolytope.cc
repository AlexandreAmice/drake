#include "drake/multibody/rational_forward_kinematics/centrally_symmetric_hpolytope.h"
#include<random>

namespace drake {
namespace multibody {

HPolyhedron GenerateRandomSeedingPolytope(const Eigen::VectorXd seed_point,
                                          const int num_unit_box_copies,
                                          const double initial_box_scale,
                                          const double gaussian_variance){
  const int nt = seed_point.rows();
  Eigen::MatrixXd C(2*nt*num_unit_box_copies, nt);
  Eigen::VectorXd d(2*nt*num_unit_box_copies);
  HPolyhedron scaled_unit_box = HPolyhedron::MakeBox(-Eigen::VectorXd::Constant(nt,1,initial_box_scale),
                                                     Eigen::VectorXd::Constant(nt,1,initial_box_scale));
  for (int i = 0; i < num_unit_box_copies; i++){
    C.block(2*nt*i, 0, 2*nt, C.cols()) = scaled_unit_box.A();
    d.block(2*nt*i, 0, 2*nt, d.cols() ) = scaled_unit_box.b();
  }
  static std::default_random_engine e(time(0));
  static std::normal_distribution <double> normal(0,gaussian_variance);
  Eigen::MatrixXd noise = Eigen::MatrixXd::Zero(C.rows(),C.cols()).unaryExpr([](double dummy){return normal( e ) +dummy;});
  HPolyhedron default_poly = HPolyhedron(C+noise, d);
  return SameDimensionalAffineTransform(Eigen::MatrixXd::Identity(nt,nt),
                                        seed_point, default_poly);
}


HPolyhedron GenerateSeedingPolytope(const Eigen::VectorXd seed_point,
                                    const int num_perm_dim, const int num_rot) {
  HPolyhedron unit_box = HPolyhedron::MakeUnitBox(seed_point.rows());

  double const scale = 0.1;
  unit_box = HPolyhedron(unit_box.A(), scale * unit_box.b());
  std::vector<Eigen::MatrixXd> SOnMembers =
      MakeKCanonicalSOnMembers(num_rot, seed_point.rows());
  std::vector<Eigen::MatrixXd> perms =
      MakeKFirstDimSwapsOfDimN(num_perm_dim, seed_point.rows());

  Eigen::MatrixXd A_new(2 * seed_point.rows() * (num_perm_dim * num_rot+1),
                        seed_point.rows());
  Eigen::VectorXd b_new(2 * seed_point.rows() * (num_perm_dim * num_rot+1));
  A_new.topRows(2 * seed_point.rows()) = unit_box.A();

  b_new.topRows(2 * seed_point.rows()) = unit_box.b() + unit_box.A() * seed_point;

  int cur_row = 2 * seed_point.rows();
  std::unique_ptr<HPolyhedron> cur_transformed_polytope_ptr{nullptr};
  for (Eigen::MatrixXd rot : SOnMembers) {
    for (Eigen::MatrixXd perm : perms) {
      cur_transformed_polytope_ptr = std::make_unique<HPolyhedron>(SameDimensionalAffineTransform(
          perm.transpose() * rot * perm, seed_point, unit_box));
      for (int i = 0; i < (*cur_transformed_polytope_ptr).A().rows(); i++) {
        A_new.row(cur_row) = (*cur_transformed_polytope_ptr).A().row(i);
        b_new.row(cur_row) = (*cur_transformed_polytope_ptr).b().row(i);
        cur_row++;
      }
    }
  }
  return HPolyhedron{A_new, b_new}.ReduceInequalities();
}

HPolyhedron SameDimensionalAffineTransform(const Eigen::MatrixXd C,
                                           const Eigen::VectorXd d,
                                           const HPolyhedron P) {
  DRAKE_DEMAND(C.rows() == d.rows());
  DRAKE_DEMAND(C.rows() == P.A().cols());

  // Computed A*pinv(C)
  Eigen::MatrixXd A_new =
      (C.transpose().fullPivLu().solve(P.A().transpose())).transpose();
  Eigen::MatrixXd b_new = P.b() + A_new * d;
  return HPolyhedron{A_new, b_new};
}

std::vector<Eigen::MatrixXd> MakeKCanonicalSOnMembers(const int k,
                                                      const int n) {
  std::vector<Eigen::MatrixXd> members;
  members.reserve(k);
  Eigen::Matrix2d rot;
  Eigen::MatrixXd SOnMember(n, n);
  SOnMember.setIdentity();
  for (int i = 0; i < k; i++) {
    for (int j = 0; j < n / 2; j++) {
      int cur_angle_ind = i * n / 2 + j;
      double cur_angle = M_PI / (n * k) * cur_angle_ind + M_PI_2 / 2;
      double a{std::cos(cur_angle)};
      double b{std::sin(cur_angle)};
      rot << a, b, -b, a;
      SOnMember.block<2, 2>(2 * j, 2 * j) = rot;
    }
    if (n % 2 == 1) {
      SOnMember(n - 1, n - 1) = 1;
    }
    members.push_back(SOnMember);
  }
  return members;
}

std::vector<Eigen::MatrixXd> MakeKFirstDimSwapsOfDimN(const int k,
                                                      const int n) {
  DRAKE_DEMAND(k <= n);
  std::vector<Eigen::MatrixXd> members;
  members.reserve(k);
  Eigen::MatrixXd perm(n, n);
  Eigen::MatrixXd c0(n, 1);
  Eigen::MatrixXd ci(n, 1);
  for (int i = 1; i < k + 1; i++) {
    perm.setIdentity();
    c0 = perm.col(0);
    ci = perm.col(i);
    perm.col(0) = ci;
    perm.col(i) = c0;
    members.push_back(perm);
  }
  return members;
}

}  // namespace multibody
}  // namespace drake