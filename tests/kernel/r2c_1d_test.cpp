#include <numsim-fft/kernel/r2c_plan_1d.h>

#include "common/random_fields.h"
#include "common/reference_dft.h"
#include "common/tolerances.h"

#include <gtest/gtest.h>

#include <complex>
#include <vector>

using namespace numsim_fft;
using numsim_fft::kernel::r2c_plan_1d;

namespace {

std::vector<std::size_t> const sizes{[] {
  std::vector<std::size_t> s;
  for (std::size_t n{1}; n <= 64; ++n)
    s.push_back(n);
  for (std::size_t n : {97u, 101u, 202u, 1000u, 1024u})
    s.push_back(n);
  return s;
}()};

/// Half spectrum (n/2+1 values) of a real signal, via the reference.
std::vector<ref::complex> reference_half_spectrum(std::vector<ref::real> const &x) {
  std::vector<ref::complex> xc(x.begin(), x.end());
  auto full{ref::dft(xc, -1)};
  full.resize(x.size() / 2 + 1);
  return full;
}

} // namespace

template <typename T> class r2c_1d : public testing::Test {};
using real_types = testing::Types<float, double, long double>;
TYPED_TEST_SUITE(r2c_1d, real_types);

TYPED_TEST(r2c_1d, forward_matches_reference_half_spectrum) {
  using T = TypeParam;
  for (auto n : sizes) {
    r2c_plan_1d<T> const plan{n};
    ASSERT_EQ(plan.spectrum_size(), n / 2 + 1);
    auto const x{test::random_vector<T>(n, static_cast<unsigned>(n))};
    std::vector<std::complex<T>> X(plan.spectrum_size());
    std::vector<std::complex<T>> scratch(plan.scratch_size(1));
    plan.forward(x.data(), 1, X.data(), 1, 1, scratch.data());
    EXPECT_LT(test::relative_error(X, reference_half_spectrum(test::convert<ref::real>(x))),
              test::tolerance<T>(n))
        << "n = " << n;
  }
}

TYPED_TEST(r2c_1d, backward_matches_reference) {
  using T = TypeParam;
  for (auto n : sizes) {
    r2c_plan_1d<T> const plan{n};
    // A Hermitian half spectrum: the transform of a random real signal.
    auto const signal{test::convert<ref::real>(test::random_vector<T>(n, 3))};
    auto const half_ref{reference_half_spectrum(signal)};
    std::vector<std::complex<T>> X(half_ref.size());
    for (std::size_t k{0}; k < X.size(); ++k)
      X[k] = {static_cast<T>(half_ref[k].real()), static_cast<T>(half_ref[k].imag())};
    // Expected: n * signal (unnormalised backward transform).
    std::vector<ref::real> expected(n);
    for (std::size_t j{0}; j < n; ++j)
      expected[j] = signal[j] * static_cast<ref::real>(n);
    std::vector<T> x(n);
    std::vector<std::complex<T>> scratch(plan.scratch_size(1));
    plan.backward(X.data(), 1, x.data(), 1, 1, scratch.data());
    EXPECT_LT(test::relative_error(x, expected), test::tolerance<T>(n, 16)) << "n = " << n;
  }
}

TYPED_TEST(r2c_1d, roundtrip_with_scale_is_identity) {
  using T = TypeParam;
  for (std::size_t n : {1u, 2u, 7u, 16u, 101u, 1000u}) {
    r2c_plan_1d<T> const plan{n};
    auto const x{test::random_vector<T>(n)};
    std::vector<std::complex<T>> X(plan.spectrum_size());
    std::vector<T> y(n);
    std::vector<std::complex<T>> scratch(plan.scratch_size(1));
    plan.forward(x.data(), 1, X.data(), 1, 1, scratch.data());
    plan.backward(X.data(), 1, y.data(), 1, 1, scratch.data(), T(1) / static_cast<T>(n));
    EXPECT_LT(test::relative_error(y, x), test::tolerance<T>(n)) << "n = " << n;
  }
}

TYPED_TEST(r2c_1d, backward_ignores_imaginary_part_of_dc_and_nyquist) {
  using T = TypeParam;
  for (std::size_t n : {8u, 9u}) {
    r2c_plan_1d<T> const plan{n};
    auto const x{test::random_vector<T>(n)};
    std::vector<std::complex<T>> X(plan.spectrum_size());
    std::vector<std::complex<T>> scratch(plan.scratch_size(1));
    plan.forward(x.data(), 1, X.data(), 1, 1, scratch.data());
    auto Y{X};
    Y.front() += std::complex<T>{0, 5};
    if (n % 2 == 0)
      Y.back() += std::complex<T>{0, -3};
    std::vector<T> a(n), b(n);
    plan.backward(X.data(), 1, a.data(), 1, 1, scratch.data());
    plan.backward(Y.data(), 1, b.data(), 1, 1, scratch.data());
    EXPECT_LT(test::relative_error(b, a), test::tolerance<T>(n)) << "n = " << n;
  }
}

TYPED_TEST(r2c_1d, strided_vector_batch) {
  using T = TypeParam;
  for (std::size_t n : {10u, 11u}) {
    std::size_t const batch{3}, in_stride{4}, out_stride{5};
    r2c_plan_1d<T> const plan{n};
    auto const in{test::random_vector<T>(n * in_stride)};
    std::vector<std::complex<T>> out(plan.spectrum_size() * out_stride, {T(9), T(9)});
    std::vector<std::complex<T>> scratch(plan.scratch_size(batch));
    plan.forward(in.data(), in_stride, out.data(), out_stride, batch, scratch.data(), T(3));
    for (std::size_t b{0}; b < batch; ++b) {
      std::vector<ref::real> line(n);
      for (std::size_t j{0}; j < n; ++j)
        line[j] = in[j * in_stride + b];
      auto expected{reference_half_spectrum(line)};
      for (auto &v : expected)
        v *= 3;
      std::vector<std::complex<T>> got(expected.size());
      for (std::size_t k{0}; k < got.size(); ++k)
        got[k] = out[k * out_stride + b];
      EXPECT_LT(test::relative_error(got, expected), test::tolerance<T>(n)) << "n=" << n;
    }
    for (std::size_t k{0}; k < plan.spectrum_size(); ++k)
      EXPECT_EQ(out[k * out_stride + batch], (std::complex<T>{T(9), T(9)}));

    // and back, into a strided real output
    std::vector<T> back(n * in_stride, T(-1));
    plan.backward(out.data(), out_stride, back.data(), in_stride, batch, scratch.data(),
                  T(1) / (T(3) * static_cast<T>(n)));
    for (std::size_t b{0}; b < batch; ++b)
      for (std::size_t j{0}; j < n; ++j)
        EXPECT_NEAR(static_cast<double>(back[j * in_stride + b]),
                    static_cast<double>(in[j * in_stride + b]),
                    static_cast<double>(test::tolerance<T>(n)));
    for (std::size_t j{0}; j < n; ++j)
      EXPECT_EQ(back[j * in_stride + batch], T(-1));
  }
}
