#ifndef NUMSIM_FFT_KERNEL_R2C_PLAN_1D_H
#define NUMSIM_FFT_KERNEL_R2C_PLAN_1D_H

#include "c2c_plan_1d.h"
#include "complex_ops.h"

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace numsim_fft::kernel {

/**
 * @brief Real-input 1D DFT of length n and its inverse, on strided vector
 * batches (same batch layout as c2c_plan_1d).
 *
 *  - forward: n reals -> the n/2+1 non-redundant values X_0 .. X_{n/2} of the
 *    Hermitian spectrum (X_{n-k} = conj X_k).
 *  - backward: n/2+1 complex values -> n reals, unnormalised: forward then
 *    backward gives n * x. The input is taken to be Hermitian: the imaginary
 *    parts of X_0 and, for even n, X_{n/2} are ignored.
 *
 * Even n: the signal is packed as z_j = x_{2j} + i x_{2j+1} and transformed
 * with a complex FFT of length n/2, then separated with
 *   X_k = E_k + w^k O_k,  E_k = (Z_k + conj Z_{h-k}) / 2,
 *   O_k = (Z_k - conj Z_{h-k}) / (2i),  w = exp(-2 pi i / n).
 * Odd n: a complex FFT of length n on the zero-imaginary signal.
 *
 * Scratch is counted in complex values. Input and output may alias.
 */
template <real_scalar T> class r2c_plan_1d {
public:
  using size_type = std::size_t;
  using real_type = T;
  using complex_type = cplx<T>;

  explicit r2c_plan_1d(size_type n)
      : _n{n}, _packed{n != 0 && n % 2 == 0},
        _fft{n == 0 ? throw std::invalid_argument("r2c_plan_1d: length must be positive")
                    : (_packed ? n / 2 : n)} {
    if (_packed) {
      _twiddles.reserve(n / 2 + 1);
      for (size_type k{0}; k <= n / 2; ++k)
        _twiddles.push_back(root_of_unity<T>(k, n));
    }
  }

  size_type size() const noexcept { return _n; }
  size_type spectrum_size() const noexcept { return _n / 2 + 1; }

  size_type scratch_size(size_type B) const noexcept {
    return _fft.size() * B + _fft.scratch_size(B);
  }

  void forward(T const *in, size_type in_stride, complex_type *out, size_type out_stride,
               size_type B, complex_type *scratch, T scale_factor = T(1)) const noexcept {
    size_type const m{_fft.size()};
    complex_type *z{scratch};
    complex_type *fft_scratch{scratch + m * B};
    if (_packed) {
      for (size_type j{0}; j < m; ++j)
        for (size_type b{0}; b < B; ++b)
          z[j * B + b] = {in[(2 * j) * in_stride + b], in[(2 * j + 1) * in_stride + b]};
      _fft.forward(z, B, z, B, B, fft_scratch);
      T const half{T(0.5) * scale_factor};
      for (size_type k{0}; k <= m; ++k) {
        size_type const k1{k == m ? 0 : k};
        size_type const k2{k == 0 ? 0 : m - k};
        for (size_type b{0}; b < B; ++b) {
          complex_type const zk{z[k1 * B + b]};
          complex_type const zc{std::conj(z[k2 * B + b])};
          complex_type const e{zk + zc};
          complex_type const o{mul_neg_i(zk - zc)};
          out[k * out_stride + b] = scale(e + cmul(o, _twiddles[k]), half);
        }
      }
    } else {
      for (size_type j{0}; j < m; ++j)
        for (size_type b{0}; b < B; ++b)
          z[j * B + b] = {in[j * in_stride + b], T(0)};
      _fft.forward(z, B, z, B, B, fft_scratch);
      for (size_type k{0}; k < spectrum_size(); ++k)
        for (size_type b{0}; b < B; ++b)
          out[k * out_stride + b] = scale(z[k * B + b], scale_factor);
    }
  }

  void backward(complex_type const *in, size_type in_stride, T *out, size_type out_stride,
                size_type B, complex_type *scratch, T scale_factor = T(1)) const noexcept {
    size_type const m{_fft.size()};
    complex_type *z{scratch};
    complex_type *fft_scratch{scratch + m * B};
    if (_packed) {
      // Z_k = (X_k + conj X_{h-k}) + i conj(w^k) (X_k - conj X_{h-k}), k < h
      for (size_type k{0}; k < m; ++k) {
        for (size_type b{0}; b < B; ++b) {
          complex_type xk{in[k * in_stride + b]};
          complex_type xc{std::conj(in[(m - k) * in_stride + b])};
          if (k == 0) { // DC and Nyquist are real
            xk = {xk.real(), T(0)};
            xc = {xc.real(), T(0)};
          }
          z[k * B + b] = (xk + xc) + mul_i(cmul(xk - xc, std::conj(_twiddles[k])));
        }
      }
      _fft.backward(z, B, z, B, B, fft_scratch);
      for (size_type j{0}; j < m; ++j)
        for (size_type b{0}; b < B; ++b) {
          out[(2 * j) * out_stride + b] = z[j * B + b].real() * scale_factor;
          out[(2 * j + 1) * out_stride + b] = z[j * B + b].imag() * scale_factor;
        }
    } else {
      // Rebuild the full Hermitian spectrum, then a complex backward FFT.
      for (size_type b{0}; b < B; ++b)
        z[b] = {in[b].real(), T(0)};
      for (size_type k{1}; k < spectrum_size(); ++k)
        for (size_type b{0}; b < B; ++b) {
          complex_type const v{in[k * in_stride + b]};
          z[k * B + b] = v;
          z[(m - k) * B + b] = std::conj(v);
        }
      _fft.backward(z, B, z, B, B, fft_scratch);
      for (size_type j{0}; j < m; ++j)
        for (size_type b{0}; b < B; ++b)
          out[j * out_stride + b] = z[j * B + b].real() * scale_factor;
    }
  }

private:
  size_type _n;
  bool _packed;
  c2c_plan_1d<T> _fft;
  std::vector<complex_type> _twiddles; // exp(-2 pi i k / n), k <= n/2
};

} // namespace numsim_fft::kernel

#endif // NUMSIM_FFT_KERNEL_R2C_PLAN_1D_H
