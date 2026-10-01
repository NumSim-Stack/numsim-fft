#ifndef NUMSIM_FFT_CORE_MDSPAN_H
#define NUMSIM_FFT_CORE_MDSPAN_H

// std::mdspan when the standard library provides it, otherwise the Kokkos
// reference implementation (selected by CMake). Use numsim_fft::md::*.

#if defined(NUMSIM_FFT_USE_KOKKOS_MDSPAN)
#include <mdspan/mdspan.hpp>
namespace numsim_fft::md {
using Kokkos::dextents;
using Kokkos::extents;
using Kokkos::layout_right;
using Kokkos::mdspan;
} // namespace numsim_fft::md
#else
#include <mdspan>
namespace numsim_fft::md {
using std::dextents;
using std::extents;
using std::layout_right;
using std::mdspan;
} // namespace numsim_fft::md
#endif

#endif // NUMSIM_FFT_CORE_MDSPAN_H
