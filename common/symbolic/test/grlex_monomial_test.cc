#include "drake/common/symbolic/grlex_monomial.h"

#include <gtest/gtest.h>

#include "drake/common/ssize.h"
#include "drake/common/test_utilities/symbolic_test_util.h"

namespace drake {
namespace symbolic {
namespace {

using test::VarLess;

class GrLexMonomialTest : public ::testing::Test {
 protected:
  const Variable var_w_{"w"};
  const Variable var_z_{"z"};
  const Variable var_y_{"y"};
  const Variable var_x_{"x"};

  std::vector<GrLexMonomial> monomials_;

  void SetUp() override {
    // These monomials are sorted in increasing order according to the
    // grev_lex order. A list of monomials this long is necessary to distinguish
    // this order from the grlex order and to check that GetNextMonomial and
    // MaybeGetPreviousMonomial are implemented correctly.
    monomials_ = {
        GrLexMonomial{
            {{var_x_, 0}, {var_y_, 0}, {var_z_, 0}, {var_w_, 0}}},  // 1.0
        GrLexMonomial{
            {{var_x_, 0}, {var_y_, 0}, {var_z_, 0}, {var_w_, 1}}},  // w
        GrLexMonomial{
            {{var_x_, 0}, {var_y_, 0}, {var_z_, 1}, {var_w_, 0}}},  // z
        GrLexMonomial{
            {{var_x_, 0}, {var_y_, 1}, {var_z_, 0}, {var_w_, 0}}},  // y
        GrLexMonomial{
            {{var_x_, 1}, {var_y_, 0}, {var_z_, 0}, {var_w_, 0}}},  // x
        GrLexMonomial{
            {{var_x_, 0}, {var_y_, 0}, {var_z_, 0}, {var_w_, 2}}},  // w^2
        GrLexMonomial{
            {{var_x_, 0}, {var_y_, 0}, {var_z_, 1}, {var_w_, 1}}},  // zw
        GrLexMonomial{
            {{var_x_, 0}, {var_y_, 0}, {var_z_, 2}, {var_w_, 0}}},  // z^2
        GrLexMonomial{
            {{var_x_, 0}, {var_y_, 1}, {var_z_, 0}, {var_w_, 1}}},  // yw
//        GrLexMonomial{
//            {{var_x_, 0}, {var_y_, 1}, {var_z_, 1}, {var_w_, 0}}},  // yz
//        GrLexMonomial{
//            {{var_x_, 0}, {var_y_, 2}, {var_z_, 0}, {var_w_, 0}}},  // y^2
//        GrLexMonomial{
//            {{var_x_, 1}, {var_y_, 0}, {var_z_, 0}, {var_w_, 1}}},  // xw
//        GrLexMonomial{
//            {{var_x_, 1}, {var_y_, 0}, {var_z_, 1}, {var_w_, 0}}},  // xz
//        GrLexMonomial{
//            {{var_x_, 1}, {var_y_, 1}, {var_z_, 0}, {var_w_, 0}}},  // xy
//        GrLexMonomial{
//            {{var_x_, 2}, {var_y_, 0}, {var_z_, 0}, {var_w_, 0}}},  // x^2
//        GrLexMonomial{
//            {{var_x_, 0}, {var_y_, 0}, {var_z_, 0}, {var_w_, 3}}},  // w^3
//        GrLexMonomial{
//            {{var_x_, 0}, {var_y_, 0}, {var_z_, 1}, {var_w_, 2}}},  // zw^2
//        GrLexMonomial{
//            {{var_x_, 0}, {var_y_, 0}, {var_z_, 2}, {var_w_, 1}}},  // z^2w
//        GrLexMonomial{
//            {{var_x_, 0}, {var_y_, 0}, {var_z_, 3}, {var_w_, 0}}},  // z^3
//        GrLexMonomial{
//            {{var_x_, 0}, {var_y_, 1}, {var_z_, 0}, {var_w_, 2}}},  // yw^2
//        GrLexMonomial{
//            {{var_x_, 0}, {var_y_, 1}, {var_z_, 1}, {var_w_, 1}}},  // yzw
//        GrLexMonomial{
//            {{var_x_, 0}, {var_y_, 1}, {var_z_, 2}, {var_w_, 0}}},  // yz^2
//        GrLexMonomial{
//            {{var_x_, 0}, {var_y_, 2}, {var_z_, 0}, {var_w_, 1}}},  // y^2w
//        GrLexMonomial{
//            {{var_x_, 0}, {var_y_, 2}, {var_z_, 1}, {var_w_, 0}}},  // y^2z
//        GrLexMonomial{
//            {{var_x_, 0}, {var_y_, 3}, {var_z_, 0}, {var_w_, 0}}},  // y^3
//        GrLexMonomial{
//            {{var_x_, 1}, {var_y_, 0}, {var_z_, 0}, {var_w_, 2}}},  // xw^2
//        GrLexMonomial{
//            {{var_x_, 1}, {var_y_, 0}, {var_z_, 1}, {var_w_, 1}}},  // xzw
//        GrLexMonomial{
//            {{var_x_, 1}, {var_y_, 0}, {var_z_, 2}, {var_w_, 0}}},  // xz^2
//        GrLexMonomial{
//            {{var_x_, 1}, {var_y_, 1}, {var_z_, 0}, {var_w_, 1}}},  // xyw
//        GrLexMonomial{
//            {{var_x_, 1}, {var_y_, 1}, {var_z_, 1}, {var_w_, 0}}},  // xyz
//        GrLexMonomial{
//            {{var_x_, 1}, {var_y_, 2}, {var_z_, 0}, {var_w_, 0}}},  // xy^2
//        GrLexMonomial{
//            {{var_x_, 2}, {var_y_, 0}, {var_z_, 0}, {var_w_, 1}}},  // x^2w
//        GrLexMonomial{
//            {{var_x_, 2}, {var_y_, 0}, {var_z_, 1}, {var_w_, 0}}},  // x^2z
//        GrLexMonomial{
//            {{var_x_, 2}, {var_y_, 1}, {var_z_, 0}, {var_w_, 0}}},  // x^2y
//        GrLexMonomial{
//            {{var_x_, 3}, {var_y_, 0}, {var_z_, 0}, {var_w_, 0}}},  // x^3
    };

    EXPECT_PRED2(VarLess, var_y_, var_x_);
    EXPECT_PRED2(VarLess, var_z_, var_y_);
    EXPECT_PRED2(VarLess, var_w_, var_z_);
  }
};

// Tests the comparison operators.
TEST_F(GrLexMonomialTest, TestComparison) {
  for (int i = 0; i < ssize(monomials_); ++i) {
    EXPECT_EQ(monomials_.at(i), monomials_.at(i));
    for (int j = i + 1; j < ssize(monomials_); ++j) {
      EXPECT_LT(monomials_.at(i), monomials_.at(j));
      EXPECT_FALSE(monomials_.at(j) < monomials_.at(j));

      EXPECT_LE(monomials_.at(i), monomials_.at(j));
      EXPECT_FALSE(monomials_.at(j) <= monomials_.at(i));

      EXPECT_GT(monomials_.at(j), monomials_.at(i));
      EXPECT_FALSE(monomials_.at(i) > monomials_.at(j));

      EXPECT_GE(monomials_.at(j), monomials_.at(i));
      EXPECT_FALSE(monomials_.at(i) >= monomials_.at(j));
    }
  }
}

 TEST_F(GrLexMonomialTest, TestGetNextMonomial) {
  for (int i = 0; i < ssize(monomials_); ++i) {
    std::cout << i << std::endl;
    std::cout << monomials_.at(i) << std::endl;
    std::unique_ptr<OrderedMonomial> next_monomial_ordered =
        monomials_.at(i).GetNextMonomial();
    GrLexMonomial* next_monomial =
        dynamic_cast<GrLexMonomial*>(next_monomial_ordered.get());
    if (i < ssize(monomials_) - 1) {
      // The list provided monomials are the first n monomials in grlex
      // order.
      EXPECT_EQ(monomials_.at(i + 1), *next_monomial);
    }
    else {
//      EXPECT_EQ(
//          *next_monomial,
//          GrLexMonomial(
//              {{var_x_, 0}, {var_y_, 0}, {var_z_, 0}, {var_w_, 4}} /*w^4*/));
    }
std::cout << std::endl;
  }
}
//
// TEST_F(GrLexMonomialTest, TestMaybeGetPreviousMonomial) {
//  // The first monomial is the 0 monomial, which always returns nullopt.
//  EXPECT_EQ(monomials_.at(0).MaybeGetPreviousMonomial(), std::nullopt);
//  for (int i = 1; i < ssize(monomials_); ++i) {
//    std::optional<std::unique_ptr<OrderedMonomial>> prev_monomial_ordered =
//        monomials_.at(i).MaybeGetPreviousMonomial();
//    // There is always a previous monomial in the grev lex order that we can
//    // get.
//    EXPECT_TRUE(prev_monomial_ordered.has_value());
//    GrLexMonomial* prev_monomial =
//        dynamic_cast<GrLexMonomial*>(prev_monomial_ordered.value().get());
//    // The list provided monomials are the first n monomials in grev_lex
//    // order.
//    EXPECT_EQ(monomials_.at(i - 1), *prev_monomial);
//  }
//}

}  // namespace
}  // namespace symbolic
}  // namespace drake
