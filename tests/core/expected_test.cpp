#include <numsim-fft/core/expected.h>

#include <gtest/gtest.h>

using namespace numsim_fft;

namespace {
expected<int, int> half(int v) {
  if (v % 2 != 0)
    return unexpected(v);
  return v / 2;
}
} // namespace

TEST(expected, value_error_and_monadic_ops) {
  EXPECT_EQ(half(8).value(), 4);
  EXPECT_EQ(half(3).error(), 3);
  EXPECT_EQ(half(8).and_then(half).value(), 2);
  EXPECT_EQ(half(8).transform([](int v) { return v + 1; }).value(), 5);
  expected<void, int> const ok{};
  EXPECT_TRUE(ok.has_value());
}
