#ifndef NUMSIM_FFT_KERNEL_C2C_PLAN_1D_H
#define NUMSIM_FFT_KERNEL_C2C_PLAN_1D_H

#include "../core/direction.h"
#include "bluestein.h"
#include "complex_ops.h"
#include "factorize.h"
#include "stockham.h"

#include <cstddef>
#include <stdexcept>

namespace numsim_fft::kernel {

/**
 * @brief Complex 1D DFT of fixed length n on strided vector batches.
 *
 * A batch holds B interleaved lines: element j of line b is at
 * `ptr[j * stride + b]` with stride >= B. Input and output may alias.
 * The transform is unnormalised; `scale` multiplies the result.
 *
 * Algorithm: mixed-radix Stockham for lengths whose prime factors are
 * <= max_direct_radix, Bluestein otherwise. The data are copied into scratch
 * (vector layout), transformed and copied back. The backward transform uses
 * conj(F(conj x)), with both conjugations fused into the copies.
 *
 * The plan is immutable after construction. Concurrent execute() calls are
 * safe with distinct scratch buffers of scratch_size(B) values.
 */
template <real_scalar T> class c2c_plan_1d {
public:
  using size_type = std::size_t;
  using value_type = cplx<T>;

  explicit c2c_plan_1d(size_type n) : _n{n}, _bluestein_used{needs_bluestein(n)} {
    if (n == 0)
      throw std::invalid_argument("c2c_plan_1d: length must be positive");
    if (_bluestein_used)
      _bluestein = bluestein<T>{n};
    else
      _stockham = stockham<T>{n};
  }

  size_type size() const noexcept { return _n; }
  bool uses_bluestein() const noexcept { return _bluestein_used; }

  size_type scratch_size(size_type B) const noexcept {
    return _bluestein_used ? _bluestein.scratch_size(B) : 2 * _n * B;
  }

  template <direction Dir>
  void execute(value_type const *in, size_type in_stride, value_type *out, size_type out_stride,
               size_type B, value_type *scratch, T scale_factor = T(1)) const noexcept {
    constexpr bool conjugate{Dir == direction::backward};
    auto const load = [&](size_type j, size_type b) {
      value_type const v{in[j * in_stride + b]};
      return conjugate ? std::conj(v) : v;
    };
    value_type const *result;
    if (_bluestein_used) {
      result = _bluestein.run(load, B, scratch);
    } else {
      value_type *a{scratch};
      for (size_type j{0}; j < _n; ++j)
        for (size_type b{0}; b < B; ++b)
          a[j * B + b] = load(j, b);
      result = _stockham.run(a, scratch + _n * B, B);
    }
    for (size_type j{0}; j < _n; ++j)
      for (size_type b{0}; b < B; ++b) {
        value_type const v{result[j * B + b]};
        out[j * out_stride + b] = scale(conjugate ? std::conj(v) : v, scale_factor);
      }
  }

  void forward(value_type const *in, size_type in_stride, value_type *out, size_type out_stride,
               size_type B, value_type *scratch, T scale_factor = T(1)) const noexcept {
    execute<direction::forward>(in, in_stride, out, out_stride, B, scratch, scale_factor);
  }

  void backward(value_type const *in, size_type in_stride, value_type *out, size_type out_stride,
                size_type B, value_type *scratch, T scale_factor = T(1)) const noexcept {
    execute<direction::backward>(in, in_stride, out, out_stride, B, scratch, scale_factor);
  }

private:
  size_type _n;
  bool _bluestein_used;
  stockham<T> _stockham;
  bluestein<T> _bluestein;
};

} // namespace numsim_fft::kernel

#endif // NUMSIM_FFT_KERNEL_C2C_PLAN_1D_H
