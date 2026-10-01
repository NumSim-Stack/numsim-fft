#ifndef NUMSIM_FFT_KERNEL_STOCKHAM_H
#define NUMSIM_FFT_KERNEL_STOCKHAM_H

#include "complex_ops.h"
#include "factorize.h"

#include <array>
#include <cstddef>
#include <vector>

namespace numsim_fft::kernel {

/// Real and imaginary parts of a batch buffer as two separate arrays
/// ("split complex"). Element j of line b is at re[j * B + b], im[j * B + b].
template <typename T> struct split {
  T *re;
  T *im;
};
template <typename T> struct const_split {
  T const *re;
  T const *im;
  constexpr const_split(split<T> s) noexcept : re{s.re}, im{s.im} {}
  constexpr const_split(T const *r, T const *i) noexcept : re{r}, im{i} {}
};

/**
 * @brief Mixed-radix Stockham autosort FFT (decimation in frequency),
 * forward direction, on vector batches in split-complex layout.
 *
 * Every butterfly runs over the B lines of the batch as its innermost loop.
 * With real and imaginary parts in separate arrays that loop is plain
 * scalar arithmetic on independent arrays, which the compiler vectorises
 * (interleaved std::complex loads are not vectorised by GCC/Clang).
 *
 * Stage with radix R on a sub-length n_s = R * m and inner count s (product
 * of the earlier radices):
 *
 *   y[q + s (R p + t)] = w_{n_s}^{p t} * sum_r x[q + s (p + r m)] w_R^{r t}
 *
 * for p < m, q < s, t < R. After the last stage the output is in natural
 * order (autosort). Stages ping-pong between two buffers; the p = 0
 * butterflies (all twiddles equal to one) skip the twiddle products.
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
   * Both hold n * B values per part. Returns the buffer holding the result.
   */
  split<T> run(split<T> a, split<T> b, size_type B) const noexcept {
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

  /// Value-type complex number for register arithmetic inside butterflies.
  struct cx {
    T r, i;
    friend constexpr cx operator+(cx a, cx b) noexcept { return {a.r + b.r, a.i + b.i}; }
    friend constexpr cx operator-(cx a, cx b) noexcept { return {a.r - b.r, a.i - b.i}; }
    friend constexpr cx operator*(cx a, cx b) noexcept {
      return {a.r * b.r - a.i * b.i, a.r * b.i + a.i * b.r};
    }
    friend constexpr cx operator*(cx a, T s) noexcept { return {a.r * s, a.i * s}; }
    constexpr cx neg_i() const noexcept { return {i, -r}; } // * (-i)
  };

  static constexpr cx to_cx(cplx<T> w) noexcept { return {w.real(), w.imag()}; }

  /// Multiplies by w, or by one when Twiddle is false (the p = 0 case).
  template <bool Twiddle> static constexpr cx tw(cx v, cx w) noexcept {
    if constexpr (Twiddle)
      return v * w;
    else
      return v;
  }

  void run_stage(stage const &st, const_split<T> x, split<T> y, size_type B) const noexcept {
    switch (st.radix) {
    case 2:
      stage_loop<2>(st, x, y, B, [](auto load, auto store, cx const *w, auto twiddle) {
        constexpr bool TW{decltype(twiddle)::value};
        cx const a0{load(0)}, a1{load(1)};
        store(0, a0 + a1);
        store(1, tw<TW>(a0 - a1, w[0]));
      });
      break;
    case 3:
      stage_loop<3>(st, x, y, B, [](auto load, auto store, cx const *w, auto twiddle) {
        constexpr bool TW{decltype(twiddle)::value};
        constexpr T half{T(0.5)};
        constexpr T sin60{T(0.866025403784438646763723170752936183L)};
        cx const a0{load(0)}, a1{load(1)}, a2{load(2)};
        cx const t1{a1 + a2};
        cx const m1{a0 - t1 * half};
        cx const m2{(a1 - a2).neg_i() * sin60};
        store(0, a0 + t1);
        store(1, tw<TW>(m1 + m2, w[0]));
        store(2, tw<TW>(m1 - m2, w[1]));
      });
      break;
    case 4:
      stage_loop<4>(st, x, y, B, [](auto load, auto store, cx const *w, auto twiddle) {
        constexpr bool TW{decltype(twiddle)::value};
        cx const a0{load(0)}, a1{load(1)}, a2{load(2)}, a3{load(3)};
        cx const s02{a0 + a2}, d02{a0 - a2}, s13{a1 + a3};
        cx const d13{(a1 - a3).neg_i()};
        store(0, s02 + s13);
        store(1, tw<TW>(d02 + d13, w[0]));
        store(2, tw<TW>(s02 - s13, w[1]));
        store(3, tw<TW>(d02 - d13, w[2]));
      });
      break;
    case 5:
      stage_loop<5>(st, x, y, B, [](auto load, auto store, cx const *w, auto twiddle) {
        constexpr bool TW{decltype(twiddle)::value};
        constexpr T c1{T(0.309016994374947424102293417182819059L)};  // cos(2pi/5)
        constexpr T c2{T(-0.809016994374947424102293417182819059L)}; // cos(4pi/5)
        constexpr T s1{T(0.951056516295153572116439333379382143L)};  // sin(2pi/5)
        constexpr T s2{T(0.587785252292473129168705954639072769L)};  // sin(4pi/5)
        cx const a0{load(0)}, a1{load(1)}, a2{load(2)}, a3{load(3)}, a4{load(4)};
        cx const t1{a1 + a4}, t2{a2 + a3}, t3{a1 - a4}, t4{a2 - a3};
        cx const p1{a0 + t1 * c1 + t2 * c2};
        cx const p2{a0 + t1 * c2 + t2 * c1};
        cx const q1{(t3 * s1 + t4 * s2).neg_i()};
        cx const q2{(t3 * s2 - t4 * s1).neg_i()};
        store(0, a0 + t1 + t2);
        store(1, tw<TW>(p1 + q1, w[0]));
        store(2, tw<TW>(p2 + q2, w[1]));
        store(3, tw<TW>(p2 - q2, w[2]));
        store(4, tw<TW>(p1 - q1, w[3]));
      });
      break;
    default:
      generic_stage(st, x, y, B);
    }
  }

  /**
   * @brief Runs `butterfly` for every (p, q) of the stage. The butterfly
   * receives `load(r)` / `store(t, value)` for the current line b; the loop
   * over b is here so that it is the innermost loop of the stage.
   */
  template <size_type R, typename Butterfly>
  void stage_loop(stage const &st, const_split<T> x, split<T> y, size_type B,
                  Butterfly &&butterfly) const noexcept {
    size_type const m{st.m}, s{st.s};
    std::array<cx, R - 1> w{};
    for (size_type p{0}; p < m; ++p) {
      for (size_type t{1}; t < R; ++t)
        w[t - 1] = to_cx(_twiddles[st.twiddle_offset + p * (R - 1) + t - 1]);
      for (size_type q{0}; q < s; ++q) {
        std::array<size_type, R> in{}, out{};
        for (size_type r{0}; r < R; ++r) {
          in[r] = (q + s * (p + r * m)) * B;
          out[r] = (q + s * (R * p + r)) * B;
        }
        auto const line = [&](auto twiddle) {
          T const *__restrict xr{x.re};
          T const *__restrict xi{x.im};
          T *__restrict yr{y.re};
          T *__restrict yi{y.im};
          for (size_type b{0}; b < B; ++b) {
            auto const load = [&](size_type r) { return cx{xr[in[r] + b], xi[in[r] + b]}; };
            auto const store = [&](size_type t, cx v) {
              yr[out[t] + b] = v.r;
              yi[out[t] + b] = v.i;
            };
            butterfly(load, store, w.data(), twiddle);
          }
        };
        if (p == 0)
          line(std::false_type{});
        else
          line(std::true_type{});
      }
    }
  }

  /// Direct O(R^2) DFT butterfly for odd primes 7 .. max_direct_radix,
  /// with the batch as the innermost loop.
  void generic_stage(stage const &st, const_split<T> x, split<T> y, size_type B) const noexcept {
    size_type const R{st.radix}, m{st.m}, s{st.s};
    cplx<T> const *roots{_roots.data() + st.roots_offset};
    for (size_type p{0}; p < m; ++p) {
      cplx<T> const *w{_twiddles.data() + st.twiddle_offset + p * (R - 1)};
      for (size_type q{0}; q < s; ++q) {
        for (size_type t{0}; t < R; ++t) {
          T *__restrict yr{y.re + (q + s * (R * p + t)) * B};
          T *__restrict yi{y.im + (q + s * (R * p + t)) * B};
          T const *x0r{x.re + (q + s * p) * B};
          T const *x0i{x.im + (q + s * p) * B};
          for (size_type b{0}; b < B; ++b) {
            yr[b] = x0r[b];
            yi[b] = x0i[b];
          }
          size_type k{0};
          for (size_type r{1}; r < R; ++r) {
            k += t;
            if (k >= R)
              k -= R;
            T const cr{roots[k].real()}, ci{roots[k].imag()};
            T const *xr{x.re + (q + s * (p + r * m)) * B};
            T const *xi{x.im + (q + s * (p + r * m)) * B};
            for (size_type b{0}; b < B; ++b) {
              yr[b] += xr[b] * cr - xi[b] * ci;
              yi[b] += xr[b] * ci + xi[b] * cr;
            }
          }
          if (t > 0 && p > 0) {
            T const wr{w[t - 1].real()}, wi{w[t - 1].imag()};
            for (size_type b{0}; b < B; ++b) {
              T const vr{yr[b]}, vi{yi[b]};
              yr[b] = vr * wr - vi * wi;
              yi[b] = vr * wi + vi * wr;
            }
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
