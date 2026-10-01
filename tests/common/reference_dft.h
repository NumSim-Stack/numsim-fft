#ifndef NUMSIM_FFT_TESTS_REFERENCE_DFT_H
#define NUMSIM_FFT_TESTS_REFERENCE_DFT_H

// Naive O(N^2) reference transforms, evaluated in long double. They define
// the conventions the library has to match:
//  - periodic: X_k = sum_n x_n exp(sign * 2 pi i n k / N), forward sign = -1,
//    unnormalised in both directions (as FFTW).
//  - r2r: FFTW REDFT00/10/01/11 and RODFT00/10/01/11, unnormalised.

#include <numsim-fft/transform/axis_kind.h>

#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <numbers>
#include <stdexcept>
#include <vector>

namespace ref {

using real = long double;
using complex = std::complex<long double>;

inline constexpr real pi = std::numbers::pi_v<long double>;

/// 1D periodic DFT with the given exponent sign (-1 forward, +1 backward).
inline std::vector<complex> dft(std::vector<complex> const &x, int sign = -1) {
  std::size_t const n{x.size()};
  // Table of the n roots of unity; (j*k) mod n indexes it exactly.
  std::vector<complex> roots(n);
  for (std::size_t m{0}; m < n; ++m) {
    real const angle{static_cast<real>(sign) * 2 * pi * static_cast<real>(m) /
                     static_cast<real>(n)};
    roots[m] = {std::cos(angle), std::sin(angle)};
  }
  std::vector<complex> y(n);
  for (std::size_t k{0}; k < n; ++k) {
    real re{0}, im{0};
    std::size_t m{0};
    for (std::size_t j{0}; j < n; ++j) {
      re += x[j].real() * roots[m].real() - x[j].imag() * roots[m].imag();
      im += x[j].real() * roots[m].imag() + x[j].imag() * roots[m].real();
      m += k;
      if (m >= n)
        m -= n;
    }
    y[k] = {re, im};
  }
  return y;
}

/// cos(pi p / q) and sin(pi p / q) with p reduced modulo 2q in integer
/// arithmetic, so the argument stays in [0, 2 pi) for any size.
inline real cos_pi(std::size_t p, std::size_t q) {
  return std::cos(pi * static_cast<real>(p % (2 * q)) / static_cast<real>(q));
}
inline real sin_pi(std::size_t p, std::size_t q) {
  return std::sin(pi * static_cast<real>(p % (2 * q)) / static_cast<real>(q));
}

/// 1D real-to-real transform in the FFTW conventions.
inline std::vector<real> r2r(std::vector<real> const &x,
                             numsim::fft::axis_kind kind) {
  using numsim::fft::axis_kind;
  std::size_t const n{x.size()};
  std::vector<real> y(n, 0);
  for (std::size_t k{0}; k < n; ++k) {
    real sum{0};
    switch (kind) {
    case axis_kind::dct1: // REDFT00, n >= 2
      if (n < 2)
        throw std::invalid_argument("dct1 needs n >= 2");
      sum = x[0] + ((k % 2 == 0) ? 1 : -1) * x[n - 1];
      for (std::size_t j{1}; j + 1 < n; ++j)
        sum += 2 * x[j] * cos_pi(j * k, n - 1);
      break;
    case axis_kind::dct2: // REDFT10
      for (std::size_t j{0}; j < n; ++j)
        sum += 2 * x[j] * cos_pi((2 * j + 1) * k, 2 * n);
      break;
    case axis_kind::dct3: // REDFT01
      sum = x[0];
      for (std::size_t j{1}; j < n; ++j)
        sum += 2 * x[j] * cos_pi(j * (2 * k + 1), 2 * n);
      break;
    case axis_kind::dct4: // REDFT11
      for (std::size_t j{0}; j < n; ++j)
        sum += 2 * x[j] * cos_pi((2 * j + 1) * (2 * k + 1), 4 * n);
      break;
    case axis_kind::dst1: // RODFT00
      for (std::size_t j{0}; j < n; ++j)
        sum += 2 * x[j] * sin_pi((j + 1) * (k + 1), n + 1);
      break;
    case axis_kind::dst2: // RODFT10
      for (std::size_t j{0}; j < n; ++j)
        sum += 2 * x[j] * sin_pi((2 * j + 1) * (k + 1), 2 * n);
      break;
    case axis_kind::dst3: // RODFT01
      sum = ((k % 2 == 0) ? 1 : -1) * x[n - 1];
      for (std::size_t j{0}; j + 1 < n; ++j)
        sum += 2 * x[j] * sin_pi((j + 1) * (2 * k + 1), 2 * n);
      break;
    case axis_kind::dst4: // RODFT11
      for (std::size_t j{0}; j < n; ++j)
        sum += 2 * x[j] * sin_pi((2 * j + 1) * (2 * k + 1), 4 * n);
      break;
    case axis_kind::periodic:
    case axis_kind::identity:
      throw std::invalid_argument("r2r: not a real-to-real kind");
    }
    y[k] = sum;
  }
  return y;
}

/// Applies a 1D transform `f` along `axis` of a row-major grid with
/// `components` interleaved values per point (the library's AoS layout).
/// `f` maps a line (std::vector<V>) to a line of possibly different length;
/// the result grid has extent `out_extent` along `axis`.
template <typename V, std::size_t Dim, typename F>
std::vector<V> along_axis(std::vector<V> const &data,
                          std::array<std::size_t, Dim> const &extents,
                          std::size_t components, std::size_t axis,
                          std::size_t out_extent, F &&f) {
  std::size_t outer{1}, inner{components};
  for (std::size_t d{0}; d < axis; ++d)
    outer *= extents[d];
  for (std::size_t d{axis + 1}; d < Dim; ++d)
    inner *= extents[d];
  std::size_t const n{extents[axis]};
  std::vector<V> out(outer * out_extent * inner);
  std::vector<V> line(n);
  for (std::size_t o{0}; o < outer; ++o) {
    for (std::size_t i{0}; i < inner; ++i) {
      for (std::size_t j{0}; j < n; ++j)
        line[j] = data[(o * n + j) * inner + i];
      auto const res{f(line)};
      for (std::size_t j{0}; j < out_extent; ++j)
        out[(o * out_extent + j) * inner + i] = res[j];
    }
  }
  return out;
}

} // namespace ref

#endif // NUMSIM_FFT_TESTS_REFERENCE_DFT_H
