#include <numsim-fft/kernel/c2c_plan_1d.h>

#include "common/random_fields.h"
#include "common/reference_dft.h"
#include "common/tolerances.h"

#include <gtest/gtest.h>

#include <complex>
#include <vector>

using namespace numsim_fft;
using numsim_fft::kernel::c2c_plan_1d;

namespace {

template <typename T>
std::vector<std::complex<T>> run(c2c_plan_1d<T> const &plan, std::vector<std::complex<T>> x,
                                 direction dir) {
  std::vector<std::complex<T>> scratch(plan.scratch_size(1));
  if (dir == direction::forward)
    plan.forward(x.data(), 1, x.data(), 1, 1, scratch.data());
  else
    plan.backward(x.data(), 1, x.data(), 1, 1, scratch.data());
  return x;
}

std::vector<std::size_t> const sizes{[] {
  std::vector<std::size_t> s;
  for (std::size_t n{1}; n <= 64; ++n)
    s.push_back(n);
  for (std::size_t n : {97u, 101u, 127u, 128u, 243u, 1000u, 1024u, 2 * 3 * 37u, 15015u})
    s.push_back(n);
  return s;
}()};

} // namespace

template <typename T> class c2c_1d : public testing::Test {};
using real_types = testing::Types<float, double, long double>;
TYPED_TEST_SUITE(c2c_1d, real_types);

TYPED_TEST(c2c_1d, forward_matches_reference) {
  using T = TypeParam;
  for (auto n : sizes) {
    c2c_plan_1d<T> const plan{n};
    auto const x{test::random_vector<std::complex<T>>(n, static_cast<unsigned>(n))};
    auto const expected{ref::dft(test::convert<ref::complex>(x), -1)};
    EXPECT_LT(test::relative_error(run(plan, x, direction::forward), expected),
              test::tolerance<T>(n))
        << "n = " << n;
  }
}

TYPED_TEST(c2c_1d, backward_matches_reference) {
  using T = TypeParam;
  for (auto n : sizes) {
    c2c_plan_1d<T> const plan{n};
    auto const x{test::random_vector<std::complex<T>>(n, static_cast<unsigned>(n + 1))};
    auto const expected{ref::dft(test::convert<ref::complex>(x), +1)};
    EXPECT_LT(test::relative_error(run(plan, x, direction::backward), expected),
              test::tolerance<T>(n))
        << "n = " << n;
  }
}

TYPED_TEST(c2c_1d, roundtrip_with_scale_is_identity) {
  using T = TypeParam;
  for (std::size_t n : {1u, 6u, 64u, 101u, 1000u}) {
    c2c_plan_1d<T> const plan{n};
    auto const x{test::random_vector<std::complex<T>>(n)};
    auto y{x};
    std::vector<std::complex<T>> scratch(plan.scratch_size(1));
    plan.forward(y.data(), 1, y.data(), 1, 1, scratch.data());
    plan.backward(y.data(), 1, y.data(), 1, 1, scratch.data(), T(1) / static_cast<T>(n));
    EXPECT_LT(test::relative_error(y, x), test::tolerance<T>(n)) << "n = " << n;
  }
}

TYPED_TEST(c2c_1d, parseval) {
  using T = TypeParam;
  for (std::size_t n : {12u, 37u, 512u}) {
    c2c_plan_1d<T> const plan{n};
    auto const x{test::random_vector<std::complex<T>>(n)};
    auto const X{run(plan, x, direction::forward)};
    long double ex{0}, eX{0};
    for (auto v : x)
      ex += test::abs2(v);
    for (auto v : X)
      eX += test::abs2(v);
    EXPECT_NEAR(static_cast<double>(eX / (static_cast<long double>(n) * ex)), 1.0,
                static_cast<double>(test::tolerance<T>(n)));
  }
}

TYPED_TEST(c2c_1d, shift_theorem) {
  using T = TypeParam;
  std::size_t const n{30};
  c2c_plan_1d<T> const plan{n};
  auto const x{test::random_vector<std::complex<T>>(n)};
  std::vector<std::complex<T>> shifted(n);
  for (std::size_t j{0}; j < n; ++j)
    shifted[(j + 1) % n] = x[j];
  auto const X{run(plan, x, direction::forward)};
  auto const S{run(plan, shifted, direction::forward)};
  std::vector<ref::complex> expected(n);
  for (std::size_t k{0}; k < n; ++k) {
    auto const a{-2 * ref::pi * static_cast<ref::real>(k) / n};
    expected[k] = ref::complex{X[k].real(), X[k].imag()} * ref::complex{std::cos(a), std::sin(a)};
  }
  EXPECT_LT(test::relative_error(S, expected), test::tolerance<T>(n, 16));
}

TYPED_TEST(c2c_1d, strided_vector_batch_out_of_place) {
  using T = TypeParam;
  // `batch` independent lines interleaved: element j of line b at j*stride+b.
  for (std::size_t n : {8u, 15u, 41u}) {
    std::size_t const batch{5}, in_stride{7}, out_stride{6};
    c2c_plan_1d<T> const plan{n};
    auto const in{test::random_vector<std::complex<T>>(n * in_stride)};
    std::complex<T> const sentinel{T(42), T(-42)};
    std::vector<std::complex<T>> out(n * out_stride, sentinel);
    std::vector<std::complex<T>> scratch(plan.scratch_size(batch));
    plan.forward(in.data(), in_stride, out.data(), out_stride, batch, scratch.data(), T(2));
    for (std::size_t b{0}; b < batch; ++b) {
      std::vector<ref::complex> line(n);
      std::vector<std::complex<T>> got(n);
      for (std::size_t j{0}; j < n; ++j) {
        line[j] = {in[j * in_stride + b].real(), in[j * in_stride + b].imag()};
        got[j] = out[j * out_stride + b];
      }
      auto expected{ref::dft(line, -1)};
      for (auto &v : expected)
        v *= 2;
      EXPECT_LT(test::relative_error(got, expected), test::tolerance<T>(n))
          << "n = " << n << " b = " << b;
    }
    // gaps between lines are untouched
    for (std::size_t j{0}; j < n; ++j)
      EXPECT_EQ(out[j * out_stride + batch], sentinel);
  }
}

TEST(c2c_1d_plan, reports_size_and_algorithm) {
  c2c_plan_1d<double> const a{1024}, b{101};
  EXPECT_EQ(a.size(), 1024u);
  EXPECT_FALSE(a.uses_bluestein());
  EXPECT_TRUE(b.uses_bluestein());
  EXPECT_GE(b.scratch_size(3), 3u * 2u * 101u);
}
