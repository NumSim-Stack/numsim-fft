#include <numsim-fft/kernel/r2r_plan_1d.h>

#include "common/random_fields.h"
#include "common/reference_dft.h"
#include "common/tolerances.h"

#include <gtest/gtest.h>

#include <complex>
#include <stdexcept>
#include <vector>

using namespace numsim_fft;
using numsim_fft::kernel::r2r_plan_1d;

namespace {

constexpr axis_kind all_r2r_kinds[]{axis_kind::dct1, axis_kind::dct2, axis_kind::dct3,
                                    axis_kind::dct4, axis_kind::dst1, axis_kind::dst2,
                                    axis_kind::dst3, axis_kind::dst4};

std::vector<std::size_t> sizes_for(axis_kind kind) {
  std::vector<std::size_t> s;
  for (std::size_t n{minimum_size(kind)}; n <= 40; ++n)
    s.push_back(n);
  for (std::size_t n : {64u, 100u, 101u, 128u, 1000u})
    s.push_back(n);
  return s;
}

char const *name(axis_kind k) {
  constexpr char const *names[]{"periodic", "dct1", "dct2", "dct3", "dct4",
                                "dst1",     "dst2", "dst3", "dst4"};
  return names[static_cast<int>(k)];
}

template <typename T>
std::vector<T> run(r2r_plan_1d<T> const &plan, std::vector<T> const &x, T scale = T(1)) {
  std::vector<T> y(x.size());
  std::vector<std::complex<T>> scratch(plan.scratch_size(1));
  plan.execute(x.data(), 1, y.data(), 1, 1, scratch.data(), scale);
  return y;
}

} // namespace

template <typename T> class r2r_1d : public testing::Test {};
using real_types = testing::Types<float, double, long double>;
TYPED_TEST_SUITE(r2r_1d, real_types);

TYPED_TEST(r2r_1d, matches_reference_definition) {
  using T = TypeParam;
  for (auto kind : all_r2r_kinds) {
    for (auto n : sizes_for(kind)) {
      r2r_plan_1d<T> const plan{n, kind};
      auto const x{test::random_vector<T>(n, static_cast<unsigned>(n))};
      auto const expected{ref::r2r(test::convert<ref::real>(x), kind)};
      EXPECT_LT(test::relative_error(run(plan, x), expected), test::tolerance<T>(n, 16))
          << name(kind) << " n = " << n;
    }
  }
}

TYPED_TEST(r2r_1d, inverse_kind_recovers_input) {
  using T = TypeParam;
  for (auto kind : all_r2r_kinds) {
    for (std::size_t n : {2u, 3u, 16u, 17u, 101u}) {
      r2r_plan_1d<T> const fwd{n, kind}, bwd{n, inverse_kind(kind)};
      auto const x{test::random_vector<T>(n)};
      auto const y{run(bwd, run(fwd, x), T(1) / static_cast<T>(logical_size(kind, n)))};
      EXPECT_LT(test::relative_error(y, x), test::tolerance<T>(n, 16))
          << name(kind) << " n = " << n;
    }
  }
}

TYPED_TEST(r2r_1d, strided_vector_batch_with_scale) {
  using T = TypeParam;
  for (auto kind : all_r2r_kinds) {
    for (std::size_t n : {6u, 7u}) {
      std::size_t const batch{3}, in_stride{5}, out_stride{4};
      r2r_plan_1d<T> const plan{n, kind};
      auto const in{test::random_vector<T>(n * in_stride)};
      std::vector<T> out(n * out_stride, T(7));
      std::vector<std::complex<T>> scratch(plan.scratch_size(batch));
      plan.execute(in.data(), in_stride, out.data(), out_stride, batch, scratch.data(), T(0.5));
      for (std::size_t b{0}; b < batch; ++b) {
        std::vector<ref::real> line(n);
        std::vector<T> got(n);
        for (std::size_t j{0}; j < n; ++j) {
          line[j] = in[j * in_stride + b];
          got[j] = out[j * out_stride + b];
        }
        auto expected{ref::r2r(line, kind)};
        for (auto &v : expected)
          v *= 0.5L;
        EXPECT_LT(test::relative_error(got, expected), test::tolerance<T>(n, 16))
            << name(kind) << " n = " << n << " b = " << b;
      }
      for (std::size_t j{0}; j < n; ++j)
        EXPECT_EQ(out[j * out_stride + batch], T(7));
    }
  }
}

TEST(r2r_1d_plan, rejects_invalid_arguments) {
  EXPECT_THROW((r2r_plan_1d<double>{1, axis_kind::dct1}), std::invalid_argument);
  EXPECT_THROW((r2r_plan_1d<double>{0, axis_kind::dct2}), std::invalid_argument);
  EXPECT_THROW((r2r_plan_1d<double>{8, axis_kind::periodic}), std::invalid_argument);
  r2r_plan_1d<double> const plan{8, axis_kind::dst3};
  EXPECT_EQ(plan.size(), 8u);
  EXPECT_EQ(plan.kind(), axis_kind::dst3);
}
