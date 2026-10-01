#ifndef NUMSIM_FFT_KERNEL_COMPLEX_OPS_H
#define NUMSIM_FFT_KERNEL_COMPLEX_OPS_H

#include "../core/scalar_traits.h"

#include <cmath>
#include <complex>
#include <cstddef>
#include <numbers>

namespace numsim::fft::kernel {

template <typename T> using cplx = std::complex<T>;

/// Plain complex product. std::complex's operator* carries the C99 Annex G
/// inf/nan recovery (a library call per product without -ffast-math).
template <typename T> constexpr cplx<T> cmul(cplx<T> a, cplx<T> b) noexcept {
  return {a.real() * b.real() - a.imag() * b.imag(),
          a.real() * b.imag() + a.imag() * b.real()};
}

/// a * (-i)
template <typename T> constexpr cplx<T> mul_neg_i(cplx<T> a) noexcept {
  return {a.imag(), -a.real()};
}

/// a * i
template <typename T> constexpr cplx<T> mul_i(cplx<T> a) noexcept {
  return {-a.imag(), a.real()};
}

template <typename T> constexpr cplx<T> scale(cplx<T> a, T s) noexcept {
  return {a.real() * s, a.imag() * s};
}

/**
 * @brief exp(-2 pi i k / n) evaluated in long double, rounded to T.
 * k is reduced modulo n first so the angle stays in [0, 2 pi).
 */
template <real_scalar T> cplx<T> root_of_unity(std::size_t k, std::size_t n) {
  long double const angle{-2.0L * std::numbers::pi_v<long double> *
                          static_cast<long double>(k % n) / static_cast<long double>(n)};
  return {static_cast<T>(std::cos(angle)), static_cast<T>(std::sin(angle))};
}

} // namespace numsim::fft::kernel

#endif // NUMSIM_FFT_KERNEL_COMPLEX_OPS_H
