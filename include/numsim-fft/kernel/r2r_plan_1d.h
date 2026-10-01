#ifndef NUMSIM_FFT_KERNEL_R2R_PLAN_1D_H
#define NUMSIM_FFT_KERNEL_R2R_PLAN_1D_H

#include "../transform/axis_kind.h"
#include "c2c_plan_1d.h"
#include "complex_ops.h"
#include "r2c_plan_1d.h"

#include <cstddef>
#include <optional>
#include <stdexcept>
#include <vector>

namespace numsim_fft::kernel {

/**
 * @brief Real-to-real 1D transforms DCT/DST I-IV (FFTW REDFT/RODFT
 * definitions, unnormalised) on strided vector batches (same batch layout as
 * c2c_plan_1d). The kind-to-inverse pairing and normalisation are given by
 * inverse_kind() and logical_size().
 *
 * Algorithms, all O(n log n) and valid for every n >= minimum_size(kind):
 *  - DCT-I:   real FFT of the even extension, length 2(n-1).
 *  - DST-I:   real FFT of the odd extension, length 2(n+1).
 *  - DCT-II:  Makhoul: even/odd reordering, real FFT of length n, twiddle.
 *  - DCT-III: inverse of Makhoul: twiddle, complex-to-real FFT of length n.
 *  - DCT-IV:  even n: complex FFT of length n/2 with pre/post twiddles;
 *             odd n: complex FFT of the zero-padded, twiddled input (2n).
 *  - DST-II:  DCT-II of (-1)^j x_j, output reversed.
 *  - DST-III: DCT-III of the reversed input, output times (-1)^k.
 *  - DST-IV:  DCT-IV of the reversed input, output times (-1)^k.
 *
 * Input and output may alias. Scratch is counted in complex values.
 *
 * Complex data: a complex batch of B lines with stride s is a real batch of
 * 2B lines with stride 2s (real and imaginary parts interleaved), so r2r
 * transforms of complex data need no separate code path.
 */
template <real_scalar T> class r2r_plan_1d {
public:
  using size_type = std::size_t;
  using complex_type = cplx<T>;

  r2r_plan_1d(size_type n, axis_kind kind) : _n{n}, _kind{kind} {
    if (!is_r2r(kind))
      throw std::invalid_argument("r2r_plan_1d: kind must be a DCT or DST");
    if (n < minimum_size(kind) || n == 0)
      throw std::invalid_argument("r2r_plan_1d: length below the minimum for this kind");

    switch (base_kind()) {
    case axis_kind::dct1:
      _r2c.emplace(2 * (n - 1));
      _real_len = 2 * (n - 1);
      _complex_len = n;
      break;
    case axis_kind::dst1:
      _r2c.emplace(2 * (n + 1));
      _real_len = 2 * (n + 1);
      _complex_len = n + 2;
      break;
    case axis_kind::dct2:
    case axis_kind::dct3:
      _r2c.emplace(n);
      _real_len = n;
      _complex_len = n / 2 + 1;
      // W^k = exp(-i pi k / (2n))
      for (size_type k{0}; k < n; ++k)
        _pre.push_back(root_of_unity<T>(k, 4 * n));
      break;
    case axis_kind::dct4:
      if (n % 2 == 0) {
        size_type const h{n / 2};
        _c2c.emplace(h);
        _complex_len = h;
        for (size_type j{0}; j < h; ++j) {
          _pre.push_back(root_of_unity<T>(4 * j + 1, 8 * n)); // exp(-i pi (4j+1)/(4n))
          _post.push_back(root_of_unity<T>(j, 2 * n));        // exp(-i pi j / n)
        }
      } else {
        _c2c.emplace(2 * n);
        _complex_len = 2 * n;
        for (size_type j{0}; j < n; ++j) {
          _pre.push_back(root_of_unity<T>(j, 4 * n));          // exp(-i pi j / (2n))
          _post.push_back(root_of_unity<T>(2 * j + 1, 8 * n)); // exp(-i pi (2j+1)/(4n))
        }
      }
      break;
    default:
      break;
    }
  }

  size_type size() const noexcept { return _n; }
  axis_kind kind() const noexcept { return _kind; }

  size_type scratch_size(size_type B) const noexcept {
    size_type const sub{_r2c ? _r2c->scratch_size(B) : _c2c->scratch_size(B)};
    return (_real_len * B + 1) / 2 + _complex_len * B + sub;
  }

  void execute(T const *in, size_type in_stride, T *out, size_type out_stride, size_type B,
               complex_type *scratch, T scale_factor = T(1)) const noexcept {
    size_type const n{_n};
    auto const sign = [](size_type i) { return (i % 2 == 0) ? T(1) : T(-1); };
    auto const direct_load = [=](size_type j, size_type b) { return in[j * in_stride + b]; };
    auto const direct_store = [=](size_type k, size_type b, T v) {
      out[k * out_stride + b] = v * scale_factor;
    };
    switch (_kind) {
    case axis_kind::dst2:
      run_base(
          [=](size_type j, size_type b) { return sign(j) * in[j * in_stride + b]; },
          [=](size_type k, size_type b, T v) { out[(n - 1 - k) * out_stride + b] = v * scale_factor; },
          B, scratch);
      break;
    case axis_kind::dst3:
    case axis_kind::dst4:
      run_base([=](size_type j, size_type b) { return in[(n - 1 - j) * in_stride + b]; },
               [=](size_type k, size_type b, T v) {
                 out[k * out_stride + b] = sign(k) * v * scale_factor;
               },
               B, scratch);
      break;
    default:
      run_base(direct_load, direct_store, B, scratch);
    }
  }

private:
  /// DCT kind computing this kind (DST-II/III/IV map to DCT-II/III/IV).
  axis_kind base_kind() const noexcept {
    switch (_kind) {
    case axis_kind::dst2:
      return axis_kind::dct2;
    case axis_kind::dst3:
      return axis_kind::dct3;
    case axis_kind::dst4:
      return axis_kind::dct4;
    default:
      return _kind;
    }
  }

  template <typename Load, typename Store>
  void run_base(Load &&load, Store &&store, size_type B, complex_type *scratch) const noexcept {
    size_type const n{_n};
    // scratch = [ real buffer | complex buffer | sub-plan scratch ]
    T *real_buf{reinterpret_cast<T *>(scratch)};
    complex_type *cbuf{scratch + (_real_len * B + 1) / 2};
    complex_type *sub{cbuf + _complex_len * B};

    switch (base_kind()) {
    case axis_kind::dct1: { // even extension [x0 .. x_{n-1}, x_{n-2} .. x1]
      size_type const N{2 * (n - 1)};
      for (size_type j{0}; j < n; ++j)
        for (size_type b{0}; b < B; ++b) {
          T const v{load(j, b)};
          real_buf[j * B + b] = v;
          if (j > 0 && j < n - 1)
            real_buf[(N - j) * B + b] = v;
        }
      _r2c->forward(real_buf, B, cbuf, B, B, sub);
      for (size_type k{0}; k < n; ++k)
        for (size_type b{0}; b < B; ++b)
          store(k, b, cbuf[k * B + b].real());
      break;
    }
    case axis_kind::dst1: { // odd extension [0, x0 .. x_{n-1}, 0, -x_{n-1} .. -x0]
      size_type const N{2 * (n + 1)};
      for (size_type b{0}; b < B; ++b) {
        real_buf[b] = T(0);
        real_buf[(n + 1) * B + b] = T(0);
      }
      for (size_type j{0}; j < n; ++j)
        for (size_type b{0}; b < B; ++b) {
          T const v{load(j, b)};
          real_buf[(j + 1) * B + b] = v;
          real_buf[(N - 1 - j) * B + b] = -v;
        }
      _r2c->forward(real_buf, B, cbuf, B, B, sub);
      for (size_type k{0}; k < n; ++k)
        for (size_type b{0}; b < B; ++b)
          store(k, b, -cbuf[(k + 1) * B + b].imag());
      break;
    }
    case axis_kind::dct2: { // Makhoul
      for (size_type j{0}; 2 * j < n; ++j)
        for (size_type b{0}; b < B; ++b)
          real_buf[j * B + b] = load(2 * j, b);
      for (size_type j{0}; 2 * j + 1 < n; ++j)
        for (size_type b{0}; b < B; ++b)
          real_buf[(n - 1 - j) * B + b] = load(2 * j + 1, b);
      _r2c->forward(real_buf, B, cbuf, B, B, sub);
      for (size_type k{0}; k < n; ++k) {
        bool const upper{k > n / 2};
        size_type const idx{upper ? n - k : k};
        for (size_type b{0}; b < B; ++b) {
          complex_type const v{upper ? std::conj(cbuf[idx * B + b]) : cbuf[idx * B + b]};
          store(k, b, T(2) * cmul(_pre[k], v).real());
        }
      }
      break;
    }
    case axis_kind::dct3: { // inverse Makhoul
      // V_k = conj(W^k) (y_k - i y_{n-k}), y_n = 0, k <= n/2
      for (size_type k{0}; k <= n / 2; ++k)
        for (size_type b{0}; b < B; ++b) {
          T const yk{load(k, b)};
          T const ynk{k == 0 ? T(0) : load(n - k, b)};
          cbuf[k * B + b] = cmul(std::conj(_pre[k]), complex_type{yk, -ynk});
        }
      _r2c->backward(cbuf, B, real_buf, B, B, sub);
      for (size_type j{0}; 2 * j < n; ++j)
        for (size_type b{0}; b < B; ++b)
          store(2 * j, b, real_buf[j * B + b]);
      for (size_type j{0}; 2 * j + 1 < n; ++j)
        for (size_type b{0}; b < B; ++b)
          store(2 * j + 1, b, real_buf[(n - 1 - j) * B + b]);
      break;
    }
    case axis_kind::dct4: {
      if (n % 2 == 0) {
        size_type const h{n / 2};
        for (size_type j{0}; j < h; ++j)
          for (size_type b{0}; b < B; ++b)
            cbuf[j * B + b] = cmul(complex_type{load(2 * j, b), load(n - 1 - 2 * j, b)}, _pre[j]);
        _c2c->forward(cbuf, B, cbuf, B, B, sub);
        for (size_type j{0}; j < h; ++j)
          for (size_type b{0}; b < B; ++b) {
            complex_type const u{cmul(cbuf[j * B + b], _post[j])};
            store(2 * j, b, T(2) * u.real());
            store(n - 1 - 2 * j, b, T(-2) * u.imag());
          }
      } else {
        for (size_type j{0}; j < n; ++j)
          for (size_type b{0}; b < B; ++b)
            cbuf[j * B + b] = scale(_pre[j], load(j, b));
        for (size_type j{n * B}; j < 2 * n * B; ++j)
          cbuf[j] = complex_type{0, 0};
        _c2c->forward(cbuf, B, cbuf, B, B, sub);
        for (size_type k{0}; k < n; ++k)
          for (size_type b{0}; b < B; ++b)
            store(k, b, T(2) * cmul(_post[k], cbuf[k * B + b]).real());
      }
      break;
    }
    default:
      break;
    }
  }

  size_type _n;
  axis_kind _kind;
  size_type _real_len{0};    // reals per line in the real buffer
  size_type _complex_len{0}; // complex values per line in the complex buffer
  std::optional<r2c_plan_1d<T>> _r2c;
  std::optional<c2c_plan_1d<T>> _c2c;
  std::vector<complex_type> _pre;
  std::vector<complex_type> _post;
};

} // namespace numsim_fft::kernel

#endif // NUMSIM_FFT_KERNEL_R2R_PLAN_1D_H
