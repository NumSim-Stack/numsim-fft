#include <numsim-fft/layout/extents.h>

#include <gtest/gtest.h>

using namespace numsim_fft;

TEST(extents, size_and_strides_are_row_major) {
  constexpr extents<3> e{4, 5, 6};
  static_assert(e.rank() == 3);
  static_assert(e.size() == 120);
  static_assert(e[0] == 4 && e[1] == 5 && e[2] == 6);
  static_assert(e.stride(0) == 30 && e.stride(1) == 6 && e.stride(2) == 1);
  static_assert(e.linear_index({1, 2, 3}) == 1 * 30 + 2 * 6 + 3);
  SUCCEED();
}

TEST(extents, deduction_guide) {
  extents e{8, 9};
  static_assert(std::is_same_v<decltype(e), extents<2>>);
  EXPECT_EQ(e.size(), 72u);
}

TEST(extents, index_roundtrip) {
  extents<3> const e{3, 4, 5};
  for (std::size_t l{0}; l < e.size(); ++l)
    EXPECT_EQ(e.linear_index(e.multi_index(l)), l);
}

TEST(extents, with_extent_replaces_one_axis) {
  extents<3> const e{8, 8, 8};
  auto const half{e.with_extent(2, 5)};
  EXPECT_EQ(half, (extents<3>{8, 8, 5}));
  EXPECT_NE(half, e);
}

TEST(extents, outer_inner_split) {
  // Number of points before/after an axis: the batching used by the plans.
  extents<3> const e{2, 3, 4};
  EXPECT_EQ(e.outer(0), 1u);
  EXPECT_EQ(e.inner(0), 12u);
  EXPECT_EQ(e.outer(1), 2u);
  EXPECT_EQ(e.inner(1), 4u);
  EXPECT_EQ(e.outer(2), 6u);
  EXPECT_EQ(e.inner(2), 1u);
}

TEST(extents, one_dimensional) {
  extents<1> const e{16};
  EXPECT_EQ(e.size(), 16u);
  EXPECT_EQ(e.stride(0), 1u);
}

TEST(extents, multi_index_on_an_empty_axis) {
  // zero extents are legal on identity axes; no division by zero
  extents<3> const e{4, 0, 3};
  EXPECT_EQ(e.size(), 0u);
  auto const idx{e.multi_index(0)};
  EXPECT_EQ(idx[1], 0u);
}
