#include "drake/multibody/rational_forward_kinematics_old/generate_monomial_basis_util.h"

#include <unordered_set>

#include <gtest/gtest.h>

namespace drake {
namespace multibody {
namespace rational_old {
void CheckGenerateMonomialBasisWithOrderUpToOne(
    const drake::symbolic::Variables& t) {
  const auto basis = GenerateMonomialBasisWithOrderUpToOne(t);
  const int basis_size_expected = 1 << static_cast<int>(t.size());
  EXPECT_EQ(basis.rows(), basis_size_expected);
  std::unordered_set<drake::symbolic::Monomial> basis_set;
  basis_set.reserve(basis_size_expected);
  for (int i = 0; i < basis.rows(); ++i) {
    for (const drake::symbolic::Variable& ti : t) {
      EXPECT_LE(basis(i).degree(ti), 1);
    }
    basis_set.insert(basis(i));
  }
  EXPECT_EQ(basis_set.size(), basis_size_expected);
}

class GenerateMonomialBasisTest : public ::testing::Test {
 protected:
  drake::symbolic::Variable t1_{"t1"};
  drake::symbolic::Variable t2_{"t2"};
  drake::symbolic::Variable t3_{"t3"};
  drake::symbolic::Variable t4_{"t4"};
};

TEST_F(GenerateMonomialBasisTest, GenerateMonomialBasisWithOrderUpToOne) {
  CheckGenerateMonomialBasisWithOrderUpToOne(drake::symbolic::Variables({t1_}));
  CheckGenerateMonomialBasisWithOrderUpToOne(
      drake::symbolic::Variables({t1_, t2_}));
  CheckGenerateMonomialBasisWithOrderUpToOne(
      drake::symbolic::Variables({t1_, t2_, t3_}));
  CheckGenerateMonomialBasisWithOrderUpToOne(
      drake::symbolic::Variables({t1_, t2_, t3_, t4_}));
}

void CheckGenerateMonomialBasisOrderAllUpToOneExceptOneUpToTwo(
    const drake::symbolic::Variables& t) {
  const auto monomial_basis =
      GenerateMonomialBasisOrderAllUpToOneExceptOneUpToTwo(t);
  const int expected_size = t.size() * (1 << (t.size() - 1)) + (1 << t.size());
  EXPECT_EQ(monomial_basis.rows(), expected_size);
  std::unordered_set<drake::symbolic::Monomial> monomial_set;
  for (int i = 0; i < expected_size; ++i) {
    monomial_set.insert(monomial_basis(i));
    int num_order_two_variables = 0;
    for (const auto& ti : t) {
      const int ti_degree = monomial_basis(i).degree(ti);
      EXPECT_LE(ti_degree, 2);
      if (ti_degree == 2) {
        num_order_two_variables++;
      }
    }
    EXPECT_LE(num_order_two_variables, 1);
  }
  EXPECT_EQ(monomial_set.size(), expected_size);
}

TEST_F(GenerateMonomialBasisTest,
       GenerateMonomialBasisOrderAllUpToOneExceptOneUpToTwo) {
  CheckGenerateMonomialBasisOrderAllUpToOneExceptOneUpToTwo(
      drake::symbolic::Variables({t1_}));
  CheckGenerateMonomialBasisOrderAllUpToOneExceptOneUpToTwo(
      drake::symbolic::Variables({t1_, t2_}));
  CheckGenerateMonomialBasisOrderAllUpToOneExceptOneUpToTwo(
      drake::symbolic::Variables({t1_, t2_, t3_}));
  CheckGenerateMonomialBasisOrderAllUpToOneExceptOneUpToTwo(
      drake::symbolic::Variables({t1_, t2_, t3_, t4_}));
}
}  // namespace rational_old
}  // namespace multibody
}  // namespace drake
