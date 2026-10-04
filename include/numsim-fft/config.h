#ifndef NUMSIM_FFT_CONFIG_H
#define NUMSIM_FFT_CONFIG_H

// Single source of the version number; CMakeLists.txt parses these lines.
#define NUMSIM_FFT_VERSION_MAJOR 0
#define NUMSIM_FFT_VERSION_MINOR 2
#define NUMSIM_FFT_VERSION_PATCH 0

// Backend availability is set by the CMake target as compile definitions:
//   NUMSIM_FFT_HAS_OPENMP, NUMSIM_FFT_HAS_HPX, NUMSIM_FFT_HAS_MPI,
//   NUMSIM_FFT_USE_KOKKOS_MDSPAN (no std::mdspan in the standard library).

#endif // NUMSIM_FFT_CONFIG_H
