#ifndef NUMSIM_FFT_TRANSFORM_AXIS_KIND_H
#define NUMSIM_FFT_TRANSFORM_AXIS_KIND_H

#include <cstddef>

namespace numsim::fft {

/**
 * @brief Transform applied along one grid axis.
 *
 * `periodic` is the ordinary DFT (complex-to-complex, or real-to-complex on
 * real data). The r2r kinds follow the FFTW definitions, all unnormalised:
 *
 * | kind | FFTW    | symmetry of the implied extension |
 * |------|---------|-----------------------------------|
 * | dct1 | REDFT00 | even about j=0 and j=n-1          |
 * | dct2 | REDFT10 | even about j=-1/2 and j=n-1/2     |
 * | dct3 | REDFT01 | even about j=0, odd about j=n     |
 * | dct4 | REDFT11 | even about j=-1/2, odd about n-1/2|
 * | dst1 | RODFT00 | odd about j=-1 and j=n            |
 * | dst2 | RODFT10 | odd about j=-1/2 and j=n-1/2      |
 * | dst3 | RODFT01 | odd about j=-1, even about j=n-1  |
 * | dst4 | RODFT11 | odd about j=-1/2, even about n-1/2|
 *
 * `identity` leaves the axis untransformed (a batch axis). It is also how
 * the distributed plans split the work into local and transposed phases.
 *
 * `periodic` is the zero value, so a value-initialised std::array of kinds
 * means "periodic on every axis".
 */
enum class axis_kind : unsigned char {
  periodic,
  dct1,
  dct2,
  dct3,
  dct4,
  dst1,
  dst2,
  dst3,
  dst4,
  identity
};

/// True for the real-to-real (DCT/DST) kinds.
constexpr bool is_r2r(axis_kind kind) noexcept {
  return kind != axis_kind::periodic && kind != axis_kind::identity;
}

/// False only for identity.
constexpr bool is_transformed(axis_kind kind) noexcept {
  return kind != axis_kind::identity;
}

/// The kind whose transform undoes `kind` (up to the factor logical_size).
constexpr axis_kind inverse_kind(axis_kind kind) noexcept {
  switch (kind) {
  case axis_kind::dct2:
    return axis_kind::dct3;
  case axis_kind::dct3:
    return axis_kind::dct2;
  case axis_kind::dst2:
    return axis_kind::dst3;
  case axis_kind::dst3:
    return axis_kind::dst2;
  default:
    return kind;
  }
}

/**
 * @brief Logical DFT size N of a transform of n points: inverse(forward(x))
 * equals N * x. This is the normalisation factor of the axis.
 */
constexpr std::size_t logical_size(axis_kind kind, std::size_t n) noexcept {
  switch (kind) {
  case axis_kind::periodic:
    return n;
  case axis_kind::identity:
    return 1;
  case axis_kind::dct1:
    return 2 * (n - 1);
  case axis_kind::dst1:
    return 2 * (n + 1);
  default:
    return 2 * n;
  }
}

/// Smallest number of points the kind is defined for.
constexpr std::size_t minimum_size(axis_kind kind) noexcept {
  switch (kind) {
  case axis_kind::identity:
    return 0;
  case axis_kind::dct1:
    return 2;
  default:
    return 1;
  }
}

} // namespace numsim::fft

#endif // NUMSIM_FFT_TRANSFORM_AXIS_KIND_H
