#ifndef NUMSIM_FFT_TRANSFORM_SPECTRAL_H
#define NUMSIM_FFT_TRANSFORM_SPECTRAL_H

#include "../core/scalar_traits.h"
#include "axis_kind.h"
#include "plan.h"

#include <array>
#include <cstddef>
#include <numbers>

namespace numsim_fft {

/**
 * @brief Wave number of spectral index k on an axis of n points with grid
 * spacing h, as the angular frequency per unit length.
 *
 *  - periodic: 2 pi k'/(n h) with the signed index k' = k for k <= n/2,
 *    k - n otherwise (also valid on the half-spectrum axis of r2c plans).
 *  - DCT/DST: the frequency of the cosine/sine basis function of index k,
 *    e.g. dct2: pi k/(n h), dst1: pi (k+1)/((n+1) h), dct3/dct4/dst3/dst4:
 *    pi (k+1/2)/(n h).
 *  - identity: 0.
 */
template <real_scalar T>
constexpr T wave_number(axis_kind kind, std::size_t k, std::size_t n, T spacing = T(1)) noexcept {
  constexpr T pi{std::numbers::pi_v<T>};
  auto const K{static_cast<T>(k)};
  auto const N{static_cast<T>(n)};
  T theta{0};
  switch (kind) {
  case axis_kind::periodic:
    theta = 2 * pi * (k <= n / 2 ? K : K - N) / N;
    break;
  case axis_kind::dct1:
    theta = pi * K / (N - 1);
    break;
  case axis_kind::dct2:
    theta = pi * K / N;
    break;
  case axis_kind::dst1:
    theta = pi * (K + 1) / (N + 1);
    break;
  case axis_kind::dst2:
    theta = pi * (K + 1) / N;
    break;
  case axis_kind::dct3:
  case axis_kind::dct4:
  case axis_kind::dst3:
  case axis_kind::dst4:
    theta = pi * (K + T(0.5)) / N;
    break;
  case axis_kind::identity:
    theta = 0;
    break;
  }
  return theta / spacing;
}

/// Index k is the Nyquist mode of a periodic axis of n points (n even,
/// k = n/2). Odd-order spectral derivatives should zero it.
constexpr bool is_nyquist(std::size_t k, std::size_t n) noexcept {
  return n % 2 == 0 && k == n / 2;
}

/// Wave vector of a point of the plan's spectral grid.
template <real_scalar T, std::size_t Dim>
std::array<T, Dim> wave_vector(plan<T, Dim> const &p, std::array<std::size_t, Dim> const &index,
                               std::array<T, Dim> const &spacing) noexcept {
  std::array<T, Dim> k{};
  for (std::size_t d{0}; d < Dim; ++d)
    k[d] = wave_number<T>(p.kinds()[d], index[d], p.physical_extents()[d], spacing[d]);
  return k;
}

} // namespace numsim_fft

#endif // NUMSIM_FFT_TRANSFORM_SPECTRAL_H
