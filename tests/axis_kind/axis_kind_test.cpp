#include <numsim-fft/transform/axis_kind.h>

#include <gtest/gtest.h>

using namespace numsim_fft;

TEST(axis_kind, r2r_classification) {
  EXPECT_FALSE(is_r2r(axis_kind::periodic));
  for (auto k : {axis_kind::dct1, axis_kind::dct2, axis_kind::dct3, axis_kind::dct4,
                 axis_kind::dst1, axis_kind::dst2, axis_kind::dst3, axis_kind::dst4})
    EXPECT_TRUE(is_r2r(k));
}

TEST(axis_kind, inverse_kinds) {
  EXPECT_EQ(inverse_kind(axis_kind::periodic), axis_kind::periodic);
  EXPECT_EQ(inverse_kind(axis_kind::dct1), axis_kind::dct1);
  EXPECT_EQ(inverse_kind(axis_kind::dct2), axis_kind::dct3);
  EXPECT_EQ(inverse_kind(axis_kind::dct3), axis_kind::dct2);
  EXPECT_EQ(inverse_kind(axis_kind::dct4), axis_kind::dct4);
  EXPECT_EQ(inverse_kind(axis_kind::dst1), axis_kind::dst1);
  EXPECT_EQ(inverse_kind(axis_kind::dst2), axis_kind::dst3);
  EXPECT_EQ(inverse_kind(axis_kind::dst3), axis_kind::dst2);
  EXPECT_EQ(inverse_kind(axis_kind::dst4), axis_kind::dst4);
}

TEST(axis_kind, logical_size) {
  EXPECT_EQ(logical_size(axis_kind::periodic, 8), 8u);
  EXPECT_EQ(logical_size(axis_kind::dct1, 8), 14u);
  EXPECT_EQ(logical_size(axis_kind::dst1, 8), 18u);
  for (auto k : {axis_kind::dct2, axis_kind::dct3, axis_kind::dct4,
                 axis_kind::dst2, axis_kind::dst3, axis_kind::dst4})
    EXPECT_EQ(logical_size(k, 8), 16u);
}

TEST(axis_kind, minimum_size) {
  EXPECT_EQ(minimum_size(axis_kind::dct1), 2u);
  EXPECT_EQ(minimum_size(axis_kind::periodic), 1u);
  EXPECT_EQ(minimum_size(axis_kind::dst1), 1u);
}

TEST(axis_kind, is_constexpr) {
  static_assert(inverse_kind(axis_kind::dct2) == axis_kind::dct3);
  static_assert(logical_size(axis_kind::dct1, 3) == 4);
  SUCCEED();
}

TEST(axis_kind, identity_leaves_the_axis_alone) {
  EXPECT_FALSE(is_r2r(axis_kind::identity));
  EXPECT_FALSE(is_transformed(axis_kind::identity));
  EXPECT_TRUE(is_transformed(axis_kind::periodic));
  EXPECT_TRUE(is_transformed(axis_kind::dst2));
  EXPECT_EQ(inverse_kind(axis_kind::identity), axis_kind::identity);
  EXPECT_EQ(logical_size(axis_kind::identity, 7), 1u);
  EXPECT_EQ(minimum_size(axis_kind::identity), 0u);
}
