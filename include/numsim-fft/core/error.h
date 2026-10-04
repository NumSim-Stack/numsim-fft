#ifndef NUMSIM_FFT_CORE_ERROR_H
#define NUMSIM_FFT_CORE_ERROR_H

#include <string_view>

namespace numsim::fft {

/// Errors reported through std::expected by plan creation and execution.
enum class error {
  zero_extent,          ///< an axis has no points
  size_below_minimum,   ///< axis shorter than its kind allows (DCT-I: n >= 2)
  periodic_axis_in_r2r, ///< real-to-real transform with a periodic axis
  no_periodic_axis,     ///< real-to-complex transform without a periodic axis
  extents_mismatch,     ///< field extents differ from the plan's
  domain_mismatch,      ///< field scalar types do not fit the plan's domain
  message_too_large,    ///< distributed: a message exceeds MPI's int count
  hooks_unsupported,    ///< point hooks on a pass whose points are not contiguous
  invalid_process_grid  ///< distributed: the process grid does not cover the ranks
};

constexpr std::string_view to_string(error e) noexcept {
  switch (e) {
  case error::zero_extent:
    return "an axis has zero extent";
  case error::size_below_minimum:
    return "axis length below the minimum of its transform kind";
  case error::periodic_axis_in_r2r:
    return "real-to-real transforms need a DCT/DST kind on every axis";
  case error::no_periodic_axis:
    return "real-to-complex transforms need at least one periodic axis";
  case error::extents_mismatch:
    return "field extents do not match the plan";
  case error::domain_mismatch:
    return "field scalar types do not match the plan's transform domain";
  case error::message_too_large:
    return "a transpose message exceeds the int count limit of MPI";
  case error::hooks_unsupported:
    return "point hooks are not supported on r2r axes of a complex transform";
  case error::invalid_process_grid:
    return "the process grid does not match the number of ranks";
  }
  return "unknown error";
}

} // namespace numsim::fft

#endif // NUMSIM_FFT_CORE_ERROR_H
