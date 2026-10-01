#ifndef NUMSIM_FFT_CORE_EXPECTED_H
#define NUMSIM_FFT_CORE_EXPECTED_H

// numsim_fft::expected / unexpected: std::expected when the standard library
// of the *consuming* compiler provides it, otherwise tl::expected. Detected
// here, not at the library's configure time: a header-only library may be
// installed with one compiler and used with another (libstdc++ 13 hides
// std::expected from Clang 18, which reports __cpp_concepts < 202002).

#include <version>

#if defined(__cpp_lib_expected) && __cpp_lib_expected >= 202202L
#include <expected>
namespace numsim_fft {
using std::expected;
using std::unexpected;
} // namespace numsim_fft
#else
#include <tl/expected.hpp>
namespace numsim_fft {
template <typename T, typename E> using expected = tl::expected<T, E>;
using tl::unexpected;
} // namespace numsim_fft
#endif

#endif // NUMSIM_FFT_CORE_EXPECTED_H
