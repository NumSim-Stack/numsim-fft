#ifndef NUMSIM_FFT_CORE_EXPECTED_H
#define NUMSIM_FFT_CORE_EXPECTED_H

// numsim_fft::expected / unexpected: std::expected when the standard library
// exposes it, otherwise tl::expected (selected by CMake). libstdc++ 13 hides
// std::expected from Clang 18, which reports __cpp_concepts < 202002.

#if defined(NUMSIM_FFT_USE_TL_EXPECTED)
#include <tl/expected.hpp>
namespace numsim_fft {
template <typename T, typename E> using expected = tl::expected<T, E>;
using tl::unexpected;
} // namespace numsim_fft
#else
#include <expected>
namespace numsim_fft {
using std::expected;
using std::unexpected;
} // namespace numsim_fft
#endif

#endif // NUMSIM_FFT_CORE_EXPECTED_H
