#ifndef NUMSIM_FFT_TESTS_RANDOM_FIELDS_H
#define NUMSIM_FFT_TESTS_RANDOM_FIELDS_H

#include <complex>
#include <cstddef>
#include <random>
#include <vector>

namespace test {

template <typename T> struct is_complex : std::false_type {};
template <typename T> struct is_complex<std::complex<T>> : std::true_type {};

/// Seeded uniform values in [-1, 1); complex types get independent parts.
template <typename T>
std::vector<T> random_vector(std::size_t n, unsigned seed = 42) {
  std::mt19937_64 gen{seed};
  std::vector<T> v(n);
  if constexpr (is_complex<T>::value) {
    using R = typename T::value_type;
    std::uniform_real_distribution<R> dist{R(-1), R(1)};
    for (auto &x : v)
      x = T{dist(gen), dist(gen)};
  } else {
    std::uniform_real_distribution<T> dist{T(-1), T(1)};
    for (auto &x : v)
      x = dist(gen);
  }
  return v;
}

/// Converts element-wise (e.g. to long double for the reference transforms).
template <typename To, typename From>
std::vector<To> convert(std::vector<From> const &in) {
  std::vector<To> out;
  out.reserve(in.size());
  for (auto const &v : in) {
    if constexpr (is_complex<To>::value && is_complex<From>::value)
      out.emplace_back(v.real(), v.imag());
    else
      out.emplace_back(static_cast<To>(v));
  }
  return out;
}

} // namespace test

#endif // NUMSIM_FFT_TESTS_RANDOM_FIELDS_H
