#ifndef NUMSIM_FFT_CORE_EXPECTED_H
#define NUMSIM_FFT_CORE_EXPECTED_H

// numsim_fft::expected / unexpected are std::expected / std::unexpected.
// std::expected is part of the NumSim toolchain baseline: GCC >= 13, or
// Clang >= 19 (libstdc++ hides <expected> from Clang 18, whose
// __cpp_concepts is still 201907). This header is the single include point,
// so a different implementation would be a local change.

#include <version>

#if !defined(__cpp_lib_expected) || __cpp_lib_expected < 202202L
#error "numsim-fft needs std::expected (C++23): use GCC >= 13 or Clang >= 19"
#endif

#include <expected>

namespace numsim_fft {
using std::expected;
using std::unexpected;
} // namespace numsim_fft

#endif // NUMSIM_FFT_CORE_EXPECTED_H
