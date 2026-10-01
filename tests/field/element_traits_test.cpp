#include <numsim-fft/field/element_traits.h>

#include <gtest/gtest.h>

#include <complex>

using namespace numsim_fft;

TEST(element_traits, scalars) {
  using tr = element_traits<double>;
  static_assert(std::is_same_v<tr::scalar_type, double>);
  static_assert(tr::components == 1);
  static_assert(std::is_same_v<tr::rebind<std::complex<double>>, std::complex<double>>);
  static_assert(field_element<std::complex<float>>);
  static_assert(!field_element<int>);
  SUCCEED();
}

TEST(element_traits, tmech_tensors) {
  static_assert(element_traits<tmech::tensor<double, 3, 2>>::components == 9);
  static_assert(element_traits<tmech::tensor<double, 2, 2>>::components == 4);
  static_assert(element_traits<tmech::tensor<float, 3, 4>>::components == 81);
  static_assert(element_traits<tmech::tensor<double, 2, 4>>::components == 16);
  using tr = element_traits<tmech::tensor<double, 3, 4>>;
  static_assert(std::is_same_v<tr::scalar_type, double>);
  static_assert(std::is_same_v<tr::rebind<std::complex<double>>,
                               tmech::tensor<std::complex<double>, 3, 4>>);
  static_assert(field_element<tmech::tensor<std::complex<double>, 3, 2>>);
  SUCCEED();
}

TEST(element_traits, rebind_to_real) {
  using c = tmech::tensor<std::complex<float>, 2, 2>;
  static_assert(std::is_same_v<element_traits<c>::rebind<float>, tmech::tensor<float, 2, 2>>);
  SUCCEED();
}
