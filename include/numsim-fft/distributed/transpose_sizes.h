#ifndef NUMSIM_FFT_DISTRIBUTED_TRANSPOSE_SIZES_H
#define NUMSIM_FFT_DISTRIBUTED_TRANSPOSE_SIZES_H

#include "slab_decomposition.h"

#include <algorithm>
#include <cstddef>
#include <limits>

namespace numsim_fft::detail {

/// a * b, saturating at the maximum of size_t instead of wrapping.
constexpr std::size_t saturating_mul(std::size_t a, std::size_t b) noexcept {
  constexpr std::size_t max{std::numeric_limits<std::size_t>::max()};
  return (a != 0 && b > max / a) ? max : a * b;
}

struct transpose_size_report {
  std::size_t largest_message;     ///< max over rank pairs of n0(p) * m1(q) * R
  std::size_t largest_local_total; ///< max over ranks of a whole local slab
};

/**
 * @brief Message and slab sizes of the slab transpose, in scalars.
 *
 * A pure function of the decomposition: every rank computes the same
 * values, so decisions based on them are collective-safe.
 */
constexpr transpose_size_report transpose_sizes(slab_decomposition rows, slab_decomposition cols,
                                                std::size_t R) noexcept {
  transpose_size_report r{0, 0};
  for (std::size_t p{0}; p < rows.parts(); ++p) {
    for (std::size_t q{0}; q < cols.parts(); ++q)
      r.largest_message = std::max(
          r.largest_message, saturating_mul(saturating_mul(rows.local_size(p), cols.local_size(q)), R));
    r.largest_local_total = std::max(
        {r.largest_local_total,
         saturating_mul(saturating_mul(rows.local_size(p), cols.global_size()), R),
         saturating_mul(saturating_mul(rows.global_size(), cols.local_size(p)), R)});
  }
  return r;
}

/**
 * @brief MPI_Alltoallv takes int counts and int element displacements. The
 * displacements reach the size of a whole local slab, so both the messages
 * and the slabs have to fit.
 */
constexpr bool transpose_fits_int(slab_decomposition rows, slab_decomposition cols,
                                  std::size_t R) noexcept {
  auto const s{transpose_sizes(rows, cols, R)};
  constexpr auto limit{static_cast<std::size_t>(std::numeric_limits<int>::max())};
  return s.largest_message <= limit && s.largest_local_total <= limit;
}

} // namespace numsim_fft::detail

#endif // NUMSIM_FFT_DISTRIBUTED_TRANSPOSE_SIZES_H
