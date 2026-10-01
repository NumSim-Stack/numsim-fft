#ifndef NUMSIM_FFT_KERNEL_BLUESTEIN_H
#define NUMSIM_FFT_KERNEL_BLUESTEIN_H

#include "complex_ops.h"
#include "factorize.h"
#include "stockham.h"

#include <cmath>
#include <cstddef>
#include <numbers>
#include <vector>

namespace numsim_fft::kernel {

/**
 * @brief Bluestein (chirp-z) forward DFT of any length n, as a cyclic
 * convolution of length M = 2^k >= 2n - 1 done with the Stockham kernel
 * (split-complex scratch, 2 M B complex values).
 *
 * With w_j = exp(-i pi j^2 / n) and jk = (j^2 + k^2 - (k - j)^2) / 2:
 *
 *   X_k = w_k * sum_j (x_j w_j) conj(w_{k-j})
 *
 * The transformed chirp kernel is precomputed, with the 1/M of the inverse
 * convolution FFT folded in. The inverse FFT is conj(FFT(conj(.))).
 */
template <real_scalar T> class bluestein {
public:
  using size_type = std::size_t;

  bluestein() = default;

  explicit bluestein(size_type n)
      : _n{n}, _m{next_power_of_two(2 * n - 1)}, _fft{_m}, _chirp(n), _kernel(_m) {
    for (size_type j{0}; j < n; ++j) {
      // j^2 mod 2n keeps the angle pi j^2 / n exact for large j.
      auto const j2{static_cast<unsigned long long>(j) * j % (2 * n)};
      long double const angle{-std::numbers::pi_v<long double> * static_cast<long double>(j2) /
                              static_cast<long double>(n)};
      _chirp[j] = {static_cast<T>(std::cos(angle)), static_cast<T>(std::sin(angle))};
    }
    // Transformed chirp kernel, with 1/M folded in (one line, split layout).
    std::vector<T> work(4 * _m, T(0));
    split<T> const a{work.data(), work.data() + _m};
    split<T> const b{work.data() + 2 * _m, work.data() + 3 * _m};
    T const inv_m{T(1) / static_cast<T>(_m)};
    auto const put = [&](size_type idx, cplx<T> v) {
      a.re[idx] = v.real() * inv_m;
      a.im[idx] = -v.imag() * inv_m; // conj
    };
    put(0, _chirp[0]);
    for (size_type d{1}; d < n; ++d) {
      put(d, _chirp[d]);
      put(_m - d, _chirp[d]);
    }
    split<T> const res{_fft.run(a, b, 1)};
    for (size_type k{0}; k < _m; ++k)
      _kernel[k] = {res.re[k], res.im[k]};
  }

  size_type size() const noexcept { return _n; }
  size_type padded_size() const noexcept { return _m; }

  /// Scratch values needed for a batch of B lines.
  size_type scratch_size(size_type B) const noexcept { return 2 * _m * B; }

  /**
   * @brief Loads the chirped, zero-padded input into scratch. `load(j, b)`
   * returns element j of line b (conjugated by the caller for backward).
   * Returns the split output: element k of line b at [k * B + b], k < n.
   */
  template <typename Load>
  const_split<T> run(Load &&load, size_type B, cplx<T> *scratch) const noexcept {
    T *s{reinterpret_cast<T *>(scratch)};
    split<T> const a{s, s + _m * B};
    split<T> const b{s + 2 * _m * B, s + 3 * _m * B};
    for (size_type j{0}; j < _n; ++j) {
      T const cr{_chirp[j].real()}, ci{_chirp[j].imag()};
      for (size_type l{0}; l < B; ++l) {
        cplx<T> const v{load(j, l)};
        a.re[j * B + l] = v.real() * cr - v.imag() * ci;
        a.im[j * B + l] = v.real() * ci + v.imag() * cr;
      }
    }
    for (size_type j{_n * B}; j < _m * B; ++j)
      a.re[j] = a.im[j] = T(0);

    split<T> const r{_fft.run(a, b, B)};
    for (size_type k{0}; k < _m; ++k) {
      T const kr{_kernel[k].real()}, ki{_kernel[k].imag()};
      T *__restrict rr{r.re + k * B};
      T *__restrict ri{r.im + k * B};
      for (size_type l{0}; l < B; ++l) { // conj(r * kernel)
        T const vr{rr[l]}, vi{ri[l]};
        rr[l] = vr * kr - vi * ki;
        ri[l] = -(vr * ki + vi * kr);
      }
    }
    split<T> const other{r.re == a.re ? b : a};
    split<T> const o{_fft.run(r, other, B)};
    for (size_type k{0}; k < _n; ++k) { // conj(o) * chirp
      T const cr{_chirp[k].real()}, ci{_chirp[k].imag()};
      T *__restrict orr{o.re + k * B};
      T *__restrict oi{o.im + k * B};
      for (size_type l{0}; l < B; ++l) {
        T const vr{orr[l]}, vi{-oi[l]};
        orr[l] = vr * cr - vi * ci;
        oi[l] = vr * ci + vi * cr;
      }
    }
    return o;
  }

private:
  size_type _n{0};
  size_type _m{0};
  stockham<T> _fft;
  std::vector<cplx<T>> _chirp;
  std::vector<cplx<T>> _kernel;
};

} // namespace numsim_fft::kernel

#endif // NUMSIM_FFT_KERNEL_BLUESTEIN_H
