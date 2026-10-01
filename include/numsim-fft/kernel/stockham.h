#ifndef NUMSIM_FFT_KERNEL_STOCKHAM_H
#define NUMSIM_FFT_KERNEL_STOCKHAM_H

#include "complex_ops.h"
#include "factorize.h"

#include <array>
#include <cstddef>
#include <vector>

namespace numsim_fft::kernel {

/**
 * @brief Mixed-radix Stockham autosort FFT (decimation in frequency),
 * forward direction, on vector batches.
 *
 * Data layout: element j of line b is at buf[j * B + b] for a batch of B
 * lines. Every butterfly therefore runs over B contiguous values, which is
 * the loop the compiler vectorises.
 *
 * Stage with radix R on a sub-length n_s = R * m and inner count s (product
 * of the earlier radices):
 *
 *   y[q + s (R p + t)] = w_{n_s}^{p t} * sum_r x[q + s (p + r m)] w_R^{r t}
 *
 * for p < m, q < s, t < R. After the last stage the output is in natural
 * order (autosort) and there is no bit reversal. Stages ping-pong between two
 * buffers.
 */
template <real_scalar T> class stockham {
public:
  using size_type = std::size_t;

  stockham() = default;

  explicit stockham(size_type n) : _n{n} {
    size_type s{1}, sub{n};
    for (auto const radix : factorize(n)) {
      stage st{radix, sub / radix, s, _twiddles.size(), 0};
      // Twiddles w_{sub}^{p t}, p < m, 1 <= t < R.
      for (size_type p{0}; p < st.m; ++p)
        for (size_type t{1}; t < radix; ++t)
          _twiddles.push_back(root_of_unity<T>(p * t, sub));
      if (radix > 5) {
        st.roots_offset = _roots.size();
        for (size_type k{0}; k < radix; ++k)
          _roots.push_back(root_of_unity<T>(k, radix));
      }
      _stages.push_back(st);
      s *= radix;
      sub /= radix;
    }
  }

  size_type size() const noexcept { return _n; }

  /**
   * @brief Forward transform of the batch in `a`, using `b` as work space.
   * Both hold n * B values. Returns the buffer holding the result (a or b).
   */
  cplx<T> *run(cplx<T> *a, cplx<T> *b, size_type B) const noexcept {
    for (auto const &st : _stages) {
      run_stage(st, a, b, B);
      std::swap(a, b);
    }
    return a;
  }

private:
  struct stage {
    size_type radix;
    size_type m;
    size_type s;
    size_type twiddle_offset;
    size_type roots_offset;
  };

  void run_stage(stage const &st, cplx<T> const *x, cplx<T> *y, size_type B) const noexcept {
    switch (st.radix) {
    case 2:
      stage_loop<2>(st, x, y, B, [](auto const &in, auto const &out, cplx<T> const *w, size_type b) {
        auto const a0{in[0][b]}, a1{in[1][b]};
        out[0][b] = a0 + a1;
        out[1][b] = cmul(a0 - a1, w[0]);
      });
      break;
    case 3:
      stage_loop<3>(st, x, y, B, [](auto const &in, auto const &out, cplx<T> const *w, size_type b) {
        constexpr T half{T(0.5)};
        constexpr T sin60{T(0.866025403784438646763723170752936183L)};
        auto const a0{in[0][b]}, a1{in[1][b]}, a2{in[2][b]};
        auto const t1{a1 + a2};
        auto const m1{a0 - scale(t1, half)};
        auto const m2{scale(mul_neg_i(a1 - a2), sin60)};
        out[0][b] = a0 + t1;
        out[1][b] = cmul(m1 + m2, w[0]);
        out[2][b] = cmul(m1 - m2, w[1]);
      });
      break;
    case 4:
      stage_loop<4>(st, x, y, B, [](auto const &in, auto const &out, cplx<T> const *w, size_type b) {
        auto const a0{in[0][b]}, a1{in[1][b]}, a2{in[2][b]}, a3{in[3][b]};
        auto const s02{a0 + a2}, d02{a0 - a2}, s13{a1 + a3};
        auto const d13{mul_neg_i(a1 - a3)};
        out[0][b] = s02 + s13;
        out[1][b] = cmul(d02 + d13, w[0]);
        out[2][b] = cmul(s02 - s13, w[1]);
        out[3][b] = cmul(d02 - d13, w[2]);
      });
      break;
    case 5:
      stage_loop<5>(st, x, y, B, [](auto const &in, auto const &out, cplx<T> const *w, size_type b) {
        constexpr T c1{T(0.309016994374947424102293417182819059L)};  // cos(2pi/5)
        constexpr T c2{T(-0.809016994374947424102293417182819059L)}; // cos(4pi/5)
        constexpr T s1{T(0.951056516295153572116439333379382143L)};  // sin(2pi/5)
        constexpr T s2{T(0.587785252292473129168705954639072769L)};  // sin(4pi/5)
        auto const a0{in[0][b]}, a1{in[1][b]}, a2{in[2][b]}, a3{in[3][b]}, a4{in[4][b]};
        auto const t1{a1 + a4}, t2{a2 + a3}, t3{a1 - a4}, t4{a2 - a3};
        auto const p1{a0 + scale(t1, c1) + scale(t2, c2)};
        auto const p2{a0 + scale(t1, c2) + scale(t2, c1)};
        auto const q1{mul_neg_i(scale(t3, s1) + scale(t4, s2))};
        auto const q2{mul_neg_i(scale(t3, s2) - scale(t4, s1))};
        out[0][b] = a0 + t1 + t2;
        out[1][b] = cmul(p1 + q1, w[0]);
        out[2][b] = cmul(p2 + q2, w[1]);
        out[3][b] = cmul(p2 - q2, w[2]);
        out[4][b] = cmul(p1 - q1, w[3]);
      });
      break;
    default:
      generic_stage(st, x, y, B);
    }
  }

  template <size_type R, typename Butterfly>
  void stage_loop(stage const &st, cplx<T> const *x, cplx<T> *y, size_type B,
                  Butterfly &&butterfly) const noexcept {
    size_type const m{st.m}, s{st.s};
    std::array<cplx<T> const *, R> in;
    std::array<cplx<T> *, R> out;
    for (size_type p{0}; p < m; ++p) {
      cplx<T> const *w{_twiddles.data() + st.twiddle_offset + p * (R - 1)};
      for (size_type q{0}; q < s; ++q) {
        for (size_type r{0}; r < R; ++r) {
          in[r] = x + (q + s * (p + r * m)) * B;
          out[r] = y + (q + s * (R * p + r)) * B;
        }
        for (size_type b{0}; b < B; ++b)
          butterfly(in, out, w, b);
      }
    }
  }

  /// Direct O(R^2) DFT butterfly for odd primes 7 .. max_direct_radix.
  void generic_stage(stage const &st, cplx<T> const *x, cplx<T> *y, size_type B) const noexcept {
    size_type const R{st.radix}, m{st.m}, s{st.s};
    cplx<T> const *roots{_roots.data() + st.roots_offset};
    std::array<cplx<T>, max_direct_radix> a;
    for (size_type p{0}; p < m; ++p) {
      cplx<T> const *w{_twiddles.data() + st.twiddle_offset + p * (R - 1)};
      for (size_type q{0}; q < s; ++q) {
        for (size_type b{0}; b < B; ++b) {
          for (size_type r{0}; r < R; ++r)
            a[r] = x[(q + s * (p + r * m)) * B + b];
          for (size_type t{0}; t < R; ++t) {
            cplx<T> sum{a[0]};
            size_type k{0};
            for (size_type r{1}; r < R; ++r) {
              k += t;
              if (k >= R)
                k -= R;
              sum += cmul(a[r], roots[k]);
            }
            y[(q + s * (R * p + t)) * B + b] = (t == 0) ? sum : cmul(sum, w[t - 1]);
          }
        }
      }
    }
  }

  size_type _n{0};
  std::vector<stage> _stages;
  std::vector<cplx<T>> _twiddles;
  std::vector<cplx<T>> _roots;
};

} // namespace numsim_fft::kernel

#endif // NUMSIM_FFT_KERNEL_STOCKHAM_H
