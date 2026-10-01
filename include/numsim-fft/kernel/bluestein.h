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
 * convolution of length M = 2^k >= 2n - 1 done with the Stockham kernel.
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
    std::vector<cplx<T>> c(2 * _m, cplx<T>{0, 0});
    T const inv_m{T(1) / static_cast<T>(_m)};
    c[0] = scale(std::conj(_chirp[0]), inv_m);
    for (size_type d{1}; d < n; ++d)
      c[d] = c[_m - d] = scale(std::conj(_chirp[d]), inv_m);
    cplx<T> const *res{_fft.run(c.data(), c.data() + _m, 1)};
    _kernel.assign(res, res + _m);
  }

  size_type size() const noexcept { return _n; }
  size_type padded_size() const noexcept { return _m; }

  /// Scratch values needed for a batch of B lines.
  size_type scratch_size(size_type B) const noexcept { return 2 * _m * B; }

  /**
   * @brief Loads the chirped, zero-padded input into scratch. `load(j, b)`
   * returns element j of line b (conjugated by the caller for backward).
   * Returns the pointer to the output: out[k * B + b] for k < n.
   */
  template <typename Load>
  cplx<T> const *run(Load &&load, size_type B, cplx<T> *scratch) const noexcept {
    cplx<T> *a{scratch};
    cplx<T> *b{scratch + _m * B};
    for (size_type j{0}; j < _n; ++j)
      for (size_type l{0}; l < B; ++l)
        a[j * B + l] = cmul(load(j, l), _chirp[j]);
    for (size_type j{_n * B}; j < _m * B; ++j)
      a[j] = cplx<T>{0, 0};

    cplx<T> *r{_fft.run(a, b, B)};
    for (size_type k{0}; k < _m; ++k)
      for (size_type l{0}; l < B; ++l)
        r[k * B + l] = std::conj(cmul(r[k * B + l], _kernel[k]));
    cplx<T> *const other{r == a ? b : a};
    cplx<T> *s{_fft.run(r, other, B)};
    for (size_type k{0}; k < _n; ++k)
      for (size_type l{0}; l < B; ++l)
        s[k * B + l] = cmul(std::conj(s[k * B + l]), _chirp[k]);
    return s;
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
