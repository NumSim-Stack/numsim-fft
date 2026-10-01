#include <numsim-fft/field/algebra.h>
#include <numsim-fft/numsim_fft.h>

#include "common/random_fields.h"

#include <gtest/gtest.h>

#include <complex>
#include <numeric>

using namespace numsim_fft;

namespace {
template <typename E, std::size_t D> field<E, D> random_field(extents<D> const &e, unsigned seed) {
  field<E, D> f{e};
  auto const v{test::random_vector<typename field<E, D>::scalar_type>(f.scalar_size(), seed)};
  std::ranges::copy(v, f.data());
  return f;
}
using tensor2 = tmech::tensor<double, 3, 2>;
} // namespace

TEST(algebra, dot_and_norm_of_real_tensor_fields) {
  extents<3> const e{5, 6, 7};
  auto const a{random_field<tensor2>(e, 1)}, b{random_field<tensor2>(e, 2)};
  double const expected{std::inner_product(a.data(), a.data() + a.scalar_size(), b.data(), 0.0)};
  EXPECT_NEAR(dot(a, b), expected, 1e-12);
  EXPECT_NEAR(norm2(a), dot(a, a), 1e-12);
  EXPECT_NEAR(norm(a), std::sqrt(dot(a, a)), 1e-12);
}

TEST(algebra, dot_of_complex_fields_is_hermitian) {
  using C = std::complex<double>;
  extents<2> const e{4, 5};
  auto const a{random_field<C>(e, 3)}, b{random_field<C>(e, 4)};
  C expected{0};
  for (std::size_t i{0}; i < a.scalar_size(); ++i)
    expected += std::conj(a.data()[i]) * b.data()[i];
  auto const d{dot(a, b)};
  EXPECT_NEAR(d.real(), expected.real(), 1e-12);
  EXPECT_NEAR(d.imag(), expected.imag(), 1e-12);
  EXPECT_NEAR(norm2(a), std::real(dot(a, a)), 1e-12); // real-valued
}

TEST(algebra, axpy_scale_and_fill_scalar) {
  extents<2> const e{3, 4};
  auto const x{random_field<tensor2>(e, 5)};
  auto y{random_field<tensor2>(e, 6)};
  auto const y0{y};
  axpy(0.5, x, y);
  for (std::size_t i{0}; i < y.scalar_size(); ++i)
    EXPECT_DOUBLE_EQ(y.data()[i], y0.data()[i] + 0.5 * x.data()[i]);
  scale(2.0, y);
  EXPECT_DOUBLE_EQ(y.data()[7], 2.0 * (y0.data()[7] + 0.5 * x.data()[7]));
  fill(y, 0.0);
  EXPECT_EQ(norm2(y), 0.0);
}

TEST(algebra, for_each_point_runs_every_point_once_and_can_use_views) {
  extents<3> const e{4, 3, 5};
  field<tensor2, 3> f{e};
  tensor2 const I{tmech::eye<double, 3, 2>()};
  for_each_point(f, [&](std::size_t p, auto view) { view = static_cast<double>(p) * I; });
  for (std::size_t p{0}; p < f.size(); ++p)
    EXPECT_DOUBLE_EQ(tmech::trace(f.point(p)), 3.0 * static_cast<double>(p));
}

TEST(algebra, executors_give_identical_reductions) {
  // the reduction order is fixed (independent of the executor), so results
  // are bitwise identical between sequential and parallel runs
  extents<3> const e{9, 8, 7};
  auto const a{random_field<tensor2>(e, 7)}, b{random_field<tensor2>(e, 8)};
  double const seq{dot(a, b, sequential_executor{})};
#if defined(NUMSIM_FFT_HAS_OPENMP)
  EXPECT_EQ(dot(a, b, openmp_executor{}), seq);
  field<tensor2, 3> y1{e}, y2{e};
  axpy(1.5, a, y1, sequential_executor{});
  axpy(1.5, a, y2, openmp_executor{});
  EXPECT_TRUE(std::ranges::equal(y1.scalars(), y2.scalars()));
#else
  EXPECT_EQ(dot(a, b), seq);
#endif
}

TEST(algebra, zero_sized_field) {
  field<tensor2, 2> const f{extents{0, 4}};
  EXPECT_EQ(norm2(f), 0.0);
}
