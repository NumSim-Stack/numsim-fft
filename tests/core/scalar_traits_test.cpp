#include <numsim-fft/core/scalar_traits.h>

#include <gtest/gtest.h>

#include <complex>

using namespace numsim::fft;

TEST(scalar_traits, complex_detection) {
  static_assert(!is_complex_v<double>);
  static_assert(is_complex_v<std::complex<float>>);
  static_assert(is_complex_v<std::complex<long double> const>);
  SUCCEED();
}

TEST(scalar_traits, real_type) {
  static_assert(std::is_same_v<real_type_t<double>, double>);
  static_assert(std::is_same_v<real_type_t<std::complex<float>>, float>);
  static_assert(std::is_same_v<complex_type_t<double>, std::complex<double>>);
  static_assert(std::is_same_v<complex_type_t<std::complex<double>>, std::complex<double>>);
  SUCCEED();
}

TEST(scalar_traits, concepts) {
  static_assert(real_scalar<float> && real_scalar<double> && real_scalar<long double>);
  static_assert(!real_scalar<int>);
  static_assert(!real_scalar<std::complex<double>>);
  static_assert(complex_scalar<std::complex<double>>);
  static_assert(!complex_scalar<double>);
  static_assert(scalar<double> && scalar<std::complex<float>>);
  static_assert(!scalar<int>);
  SUCCEED();
}
