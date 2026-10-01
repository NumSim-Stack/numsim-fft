#ifndef NUMSIM_FFT_TESTS_REFERENCE_ND_H
#define NUMSIM_FFT_TESTS_REFERENCE_ND_H

// n-D references built from the 1D ones, on the library's AoS layout
// (row-major points, `components` interleaved values per point).

#include "reference_dft.h"

#include <array>
#include <cstddef>
#include <vector>

namespace ref {

/// Forward n-D transform of complex data: DFT (sign -1) on periodic axes,
/// the r2r kind applied to real and imaginary parts on the other axes.
template <std::size_t Dim>
std::vector<complex> forward_nd(std::vector<complex> data, std::array<std::size_t, Dim> const &ext,
                                std::size_t components,
                                std::array<numsim_fft::axis_kind, Dim> const &kinds,
                                int sign = -1) {
  for (std::size_t axis{0}; axis < Dim; ++axis) {
    auto const kind{kinds[axis]};
    if (kind == numsim_fft::axis_kind::identity)
      continue;
    data = along_axis(data, ext, components, axis, ext[axis],
                      [&](std::vector<complex> const &line) {
                        if (kind == numsim_fft::axis_kind::periodic)
                          return dft(line, sign);
                        std::vector<real> re(line.size()), im(line.size());
                        for (std::size_t j{0}; j < line.size(); ++j) {
                          re[j] = line[j].real();
                          im[j] = line[j].imag();
                        }
                        auto const yr{r2r(re, kind)}, yi{r2r(im, kind)};
                        std::vector<complex> out(line.size());
                        for (std::size_t j{0}; j < line.size(); ++j)
                          out[j] = {yr[j], yi[j]};
                        return out;
                      });
  }
  return data;
}

/// Keeps indices 0 .. new_extent-1 along `axis`.
template <typename V, std::size_t Dim>
std::vector<V> crop(std::vector<V> const &data, std::array<std::size_t, Dim> const &ext,
                    std::size_t components, std::size_t axis, std::size_t new_extent) {
  return along_axis(data, ext, components, axis, new_extent, [](std::vector<V> const &line) {
    return line;
  });
}

} // namespace ref

#endif // NUMSIM_FFT_TESTS_REFERENCE_ND_H
