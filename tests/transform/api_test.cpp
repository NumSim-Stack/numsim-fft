#include <numsim-fft/transform/api.h>

#include "common/random_fields.h"
#include "common/tolerances.h"

#include <gtest/gtest.h>

#include <complex>

using namespace numsim::fft;

namespace {
template <typename E, std::size_t D> field<E, D> random_field(extents<D> const &e) {
  field<E, D> f{e};
  auto const v{test::random_vector<typename field<E, D>::scalar_type>(f.scalar_size(), 3)};
  std::ranges::copy(v, f.data());
  return f;
}
} // namespace

TEST(api, fft_ifft_roundtrip) {
  using C = std::complex<double>;
  auto const x{random_field<C>(extents{6, 7})};
  auto const X{fft(x).value()};
  EXPECT_EQ(X.extents(), x.extents());
  auto const y{ifft(X).value()};
  EXPECT_LT(test::relative_error(y.scalars(), x.scalars()), 1e-14L);
}

TEST(api, rfft_irfft_roundtrip_with_tensors) {
  using tensor2 = tmech::tensor<double, 3, 2>;
  auto const x{random_field<tensor2>(extents{4, 6, 5})};
  auto const X{rfft(x).value()};
  static_assert(std::is_same_v<std::remove_cvref_t<decltype(X)>::element_type,
                               tmech::tensor<std::complex<double>, 3, 2>>);
  EXPECT_EQ(X.extents(), (extents{4, 6, 3}));
  auto const y{irfft(X, x.extents()).value()};
  EXPECT_LT(test::relative_error(y.scalars(), x.scalars()), 1e-14L);
}

TEST(api, r2r_roundtrip) {
  auto const x{random_field<double>(extents{5, 8})};
  std::array<axis_kind, 2> const kinds{axis_kind::dct2, axis_kind::dst1};
  auto const X{r2r(x, kinds).value()};
  auto const y{inverse_r2r(X, kinds).value()};
  EXPECT_LT(test::relative_error(y.scalars(), x.scalars()), 1e-14L);
}

TEST(api, errors_propagate) {
  field<double, 2> const x{extents{4, 1}};
  EXPECT_EQ(r2r(x, {axis_kind::dct1, axis_kind::dct1}).error(), error::size_below_minimum);
}
