// Sanity checks of the naive reference transforms against closed forms, so
// that the library tests compare against something known to be right.

#include "common/reference_dft.h"
#include "common/random_fields.h"
#include "common/tolerances.h"

#include <gtest/gtest.h>

using numsim_fft::axis_kind;

namespace {
constexpr long double tight{1e-15L};
}

TEST(reference_dft, delta_transforms_to_ones) {
  std::vector<ref::complex> x(7, 0);
  x[0] = 1;
  std::vector<ref::complex> const ones(7, 1);
  EXPECT_LT(test::relative_error(ref::dft(x), ones), tight);
}

TEST(reference_dft, single_mode_transforms_to_spike) {
  std::size_t const n{12}, m{5};
  std::vector<ref::complex> x(n);
  for (std::size_t j{0}; j < n; ++j) {
    auto const a{2 * ref::pi * static_cast<ref::real>(j * m) / n};
    x[j] = {std::cos(a), std::sin(a)};
  }
  std::vector<ref::complex> expected(n, 0);
  expected[m] = static_cast<ref::real>(n);
  EXPECT_LT(test::relative_error(ref::dft(x, -1), expected), tight);
}

TEST(reference_dft, forward_backward_is_n_times_identity) {
  auto const x{test::convert<ref::complex>(test::random_vector<std::complex<double>>(15))};
  auto y{ref::dft(ref::dft(x, -1), +1)};
  for (auto &v : y)
    v /= 15;
  EXPECT_LT(test::relative_error(y, x), tight);
}

TEST(reference_r2r, dct2_of_constant) {
  std::size_t const n{6};
  std::vector<ref::real> const x(n, 3);
  std::vector<ref::real> expected(n, 0);
  expected[0] = 2 * n * 3;
  EXPECT_LT(test::relative_error(ref::r2r(x, axis_kind::dct2), expected), tight);
}

struct r2r_pair {
  axis_kind forward;
  axis_kind backward;
  // Logical size N such that backward(forward(x)) = N * x.
  std::size_t (*scale)(std::size_t);
};

class reference_r2r_pairs : public testing::TestWithParam<r2r_pair> {};

TEST_P(reference_r2r_pairs, inverse_pair_is_scaled_identity) {
  auto const [fwd, bwd, scale] = GetParam();
  for (std::size_t n : {2u, 3u, 5u, 8u}) {
    auto const x{test::convert<ref::real>(test::random_vector<double>(n, 7))};
    auto y{ref::r2r(ref::r2r(x, fwd), bwd)};
    for (auto &v : y)
      v /= static_cast<ref::real>(scale(n));
    EXPECT_LT(test::relative_error(y, x), tight) << "n = " << n;
  }
}

INSTANTIATE_TEST_SUITE_P(
    all_kinds, reference_r2r_pairs,
    testing::Values(
        r2r_pair{axis_kind::dct1, axis_kind::dct1, [](std::size_t n) { return 2 * (n - 1); }},
        r2r_pair{axis_kind::dct2, axis_kind::dct3, [](std::size_t n) { return 2 * n; }},
        r2r_pair{axis_kind::dct3, axis_kind::dct2, [](std::size_t n) { return 2 * n; }},
        r2r_pair{axis_kind::dct4, axis_kind::dct4, [](std::size_t n) { return 2 * n; }},
        r2r_pair{axis_kind::dst1, axis_kind::dst1, [](std::size_t n) { return 2 * (n + 1); }},
        r2r_pair{axis_kind::dst2, axis_kind::dst3, [](std::size_t n) { return 2 * n; }},
        r2r_pair{axis_kind::dst3, axis_kind::dst2, [](std::size_t n) { return 2 * n; }},
        r2r_pair{axis_kind::dst4, axis_kind::dst4, [](std::size_t n) { return 2 * n; }}));

TEST(reference_along_axis, picks_strided_lines_with_components) {
  // 2 x 3 grid, 2 components; reverse every line along axis 0.
  std::array<std::size_t, 2> const ext{2, 3};
  std::vector<int> data(12);
  for (int i{0}; i < 12; ++i)
    data[static_cast<std::size_t>(i)] = i;
  auto const out{ref::along_axis(data, ext, 2, 0, 2, [](std::vector<int> line) {
    std::ranges::reverse(line);
    return line;
  })};
  // point (0,1) component 1 lives at index (0*3+1)*2+1 = 3; after reversing
  // axis 0 it holds the value of point (1,1) component 1: (1*3+1)*2+1 = 9.
  EXPECT_EQ(out[3], 9);
  EXPECT_EQ(out[9], 3);
}
