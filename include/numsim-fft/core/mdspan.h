#ifndef NUMSIM_FFT_CORE_MDSPAN_H
#define NUMSIM_FFT_CORE_MDSPAN_H

// std::mdspan when the consuming standard library provides it, otherwise the
// Kokkos reference implementation (either its current <mdspan/mdspan.hpp>,
// namespace Kokkos, or the older <experimental/mdspan>, namespace
// std::experimental). Use numsim::fft::md::*.

#include <version>

#if defined(__cpp_lib_mdspan) && __cpp_lib_mdspan >= 202207L
#include <mdspan>
namespace numsim::fft::md {
using std::dextents;
using std::extents;
using std::layout_right;
using std::mdspan;
} // namespace numsim::fft::md
#elif __has_include(<mdspan/mdspan.hpp>)
#include <mdspan/mdspan.hpp>
namespace numsim::fft::md {
using Kokkos::dextents;
using Kokkos::extents;
using Kokkos::layout_right;
using Kokkos::mdspan;
} // namespace numsim::fft::md
#elif __has_include(<experimental/mdspan>)
#include <experimental/mdspan>
namespace numsim::fft::md {
using std::experimental::dextents;
using std::experimental::extents;
using std::experimental::layout_right;
using std::experimental::mdspan;
} // namespace numsim::fft::md
#else
#error "numsim-fft: no mdspan implementation found (std::mdspan or kokkos/mdspan)"
#endif

#endif // NUMSIM_FFT_CORE_MDSPAN_H
