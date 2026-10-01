#ifndef NUMSIM_FFT_TESTS_TOLERANCES_H
#define NUMSIM_FFT_TESTS_TOLERANCES_H

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <ranges>

namespace test {

/// Relative L2 tolerance for a transform of total length n in precision T:
/// c * eps * log2(n), with a floor so that tiny sizes are not over-strict.
template <typename T> long double tolerance(std::size_t n, long double c = 8) {
  long double const eps{std::numeric_limits<T>::epsilon()};
  long double const logn{std::max<long double>(
      1.0L, std::log2(static_cast<long double>(std::max<std::size_t>(n, 2))))};
  return c * eps * logn;
}

template <typename T> long double abs2(T const &v) {
  return static_cast<long double>(std::norm(v));
}

/// ||a - b||_2 / max(||b||_2, tiny). Element types may differ in precision
/// (a in T, b in long double); both are promoted to long double.
template <std::ranges::range A, std::ranges::range B>
long double relative_error(A const &a, B const &b) {
  long double num{0}, den{0};
  auto ib{std::ranges::begin(b)};
  for (auto const &va : a) {
    auto const vb{*ib++};
    if constexpr (requires { va.real(); }) {
      std::complex<long double> const da{va.real(), va.imag()};
      std::complex<long double> const db{vb.real(), vb.imag()};
      num += std::norm(da - db);
      den += std::norm(db);
    } else {
      long double const d{static_cast<long double>(va) -
                          static_cast<long double>(vb)};
      num += d * d;
      den += static_cast<long double>(vb) * static_cast<long double>(vb);
    }
  }
  return std::sqrt(num) /
         std::max(std::sqrt(den), std::numeric_limits<long double>::min());
}

} // namespace test

#endif // NUMSIM_FFT_TESTS_TOLERANCES_H
