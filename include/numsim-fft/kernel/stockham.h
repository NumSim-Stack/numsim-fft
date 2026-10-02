#ifndef NUMSIM_FFT_KERNEL_STOCKHAM_H
#define NUMSIM_FFT_KERNEL_STOCKHAM_H

#include "complex_ops.h"
#include "factorize.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <vector>

namespace numsim::fft::kernel {

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
 * forward direction, on vector batches.
 *
 * Every butterfly runs over the B lines of the batch as its innermost loop.
 * Each input and output row is a named pointer (the outputs restrict), so
 * that loop is plain arithmetic on independent arrays which GCC and Clang
 * vectorise.
 *
 * Stage with radix R on a sub-length n_s = R * m and inner count s (product
 * of the earlier radices):
 *
 *   y[q + s (R p + t)] = w_{n_s}^{p t} * sum_r x[q + s (p + r m)] w_R^{r t}
 *
 * for p < m, q < s, t < R. After the last stage the output is in natural
 * order (autosort). Stages ping-pong between two split-complex scratch
 * buffers; the p = 0 butterflies (all twiddles equal to one) skip the
 * twiddle products.
 *
 * Rows are addressed through a `layout`: a real and an imaginary base
 * pointer, a row stride and a compile-time element stride (1 for split
 * scratch, 2 for interleaved std::complex data). The first stage reads the
 * interleaved input and the last stage writes the interleaved output
 * directly, so no layout-conversion passes are needed. The backward
 * transform swaps real and imaginary parts on input and output, which is
 * conj(F(conj x)) without a sign change, since swap(z) = i conj(z).
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
      if (!has_butterfly(radix)) {
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

  /// Scratch (in values of cplx<T>) for a batch of B lines.
  size_type scratch_size(size_type B) const noexcept { return 2 * _n * B; }

  /// Row addressing: row j, line b of the real part is re[j * row + S * b]
  /// (S = 1 for split, S = 2 for interleaved data).
  template <typename P, size_type S> struct layout {
    P re;
    P im;
    size_type row;
  };

  static layout<T const *, 1> from(split<T> a, size_type B) noexcept {
    return {a.re, a.im, B};
  }
  static layout<T *, 1> to(split<T> a, size_type B) noexcept {
    return {a.re, a.im, B};
  }

  /**
   * @brief Transform of B interleaved lines: element j of line b is at
   * in[j * in_stride + b], likewise for the output. The result is scaled
   * by `scale`; `backward` selects the inverse (unnormalised) transform.
   * Input and output may be the same memory.
   */
  void execute(cplx<T> const *in, size_type in_stride, cplx<T> *out,
               size_type out_stride, size_type B, cplx<T> *scratch, T scale,
               bool backward) const noexcept {
    T const *const ib{reinterpret_cast<T const *>(in)};
    T *const ob{reinterpret_cast<T *>(out)};
    transform(layout<T const *, 2>{ib, ib + 1, 2 * in_stride},
              layout<T *, 2>{ob, ob + 1, 2 * out_stride}, B, scratch, scale,
              backward);
  }

  /**
   * @brief Transform between arbitrary row layouts (split or interleaved,
   * any row stride). `scratch` holds scratch_size(B) values. Input and
   * output may overlap.
   */
  template <typename PX, size_type XS, size_type YS>
  void transform(layout<PX, XS> x, layout<T *, YS> y, size_type B,
                 cplx<T> *scratch, T scale, bool backward) const noexcept {
    if (backward) { // swap(z) = i conj(z): swap(F(swap x)) = conj(F(conj x))
      std::swap(x.re, x.im);
      std::swap(y.re, y.im);
    }
    T *const s{reinterpret_cast<T *>(scratch)};
    split<T> a{s, s + _n * B}, b{s + 2 * _n * B, s + 3 * _n * B};
    size_type const S{_stages.size()};
    if (S == 0) {
      copy_row<XS, YS>(B, x.re, x.im, y.re, y.im, scale);
      return;
    }
    if (S == 1) {
      // A single stage reads and writes the same rows: go through scratch
      // when the ranges overlap.
      if (overlap(x, y, B)) {
        for (size_type j{0}; j < _n; ++j)
          copy_row<XS, 1>(B, x.re + j * x.row, x.im + j * x.row, a.re + j * B,
                          a.im + j * B, T(1));
        run_stage<true>(_stages[0], from(a, B), y, B, scale);
      } else {
        run_stage<true>(_stages[0], x, y, B, scale);
      }
      return;
    }
    run_stage<false>(_stages[0], x, to(a, B), B, T(1));
    for (size_type k{1}; k + 1 < S; ++k) {
      run_stage<false>(_stages[k], from(a, B), to(b, B), B, T(1));
      std::swap(a, b);
    }
    run_stage<true>(_stages[S - 1], from(a, B), y, B, scale);
  }

  /**
   * @brief Forward transform of the split-complex batch in `a`, using `b`
   * as work space. Both hold n * B values per part. Returns the buffer
   * holding the result.
   */
  split<T> run(split<T> a, split<T> b, size_type B) const noexcept {
    for (auto const &st : _stages) {
      run_stage<false>(st, from(a, B), to(b, B), B, T(1));
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

  /// Row pointers of one butterfly: R inputs and R outputs.
  template <size_type R> struct rows {
    std::array<T const *, R> xr, xi;
    std::array<T *, R> yr, yi;
  };

  static constexpr bool has_butterfly(size_type radix) noexcept {
    return radix == 2 || radix == 3 || radix == 4 || radix == 5 || radix == 8;
  }

  template <size_type XS, size_type YS>
  static void copy_row(size_type B, T const *xr, T const *xi, T *__restrict yr,
                       T *__restrict yi, T scale) noexcept {
    for (size_type b{0}; b < B; ++b) {
      yr[YS * b] = xr[XS * b] * scale;
      yi[YS * b] = xi[XS * b] * scale;
    }
  }

  template <typename PX, size_type XS, size_type YS>
  bool overlap(layout<PX, XS> x, layout<T *, YS> y,
               size_type B) const noexcept {
    T const *const xlo{std::min(x.re, x.im)};
    T const *const xhi{std::max(x.re, x.im) + (_n - 1) * x.row + XS * (B - 1) +
                       1};
    T const *const ylo{std::min(y.re, y.im)};
    T const *const yhi{std::max(y.re, y.im) + (_n - 1) * y.row + YS * (B - 1) +
                       1};
    return xlo < yhi && ylo < xhi;
  }

  template <bool Last, typename PX, size_type XS, size_type YS>
  void run_stage(stage const &st, layout<PX, XS> x, layout<T *, YS> y,
                 size_type B, T scale) const noexcept {
    switch (st.radix) {
    case 2:
      stage_loop<2>(
          st, x, y, B,
          [=](size_type B, rows<2> const &o, cplx<T> const *w, auto tw) {
            radix2<decltype(tw)::value, XS, YS, Last>(B, o, w, scale);
          });
      break;
    case 3:
      stage_loop<3>(
          st, x, y, B,
          [=](size_type B, rows<3> const &o, cplx<T> const *w, auto tw) {
            radix3<decltype(tw)::value, XS, YS, Last>(B, o, w, scale);
          });
      break;
    case 4:
      stage_loop<4>(
          st, x, y, B,
          [=](size_type B, rows<4> const &o, cplx<T> const *w, auto tw) {
            radix4<decltype(tw)::value, XS, YS, Last>(B, o, w, scale);
          });
      break;
    case 5:
      stage_loop<5>(
          st, x, y, B,
          [=](size_type B, rows<5> const &o, cplx<T> const *w, auto tw) {
            radix5<decltype(tw)::value, XS, YS, Last>(B, o, w, scale);
          });
      break;
    case 8:
      stage_loop<8>(
          st, x, y, B,
          [=](size_type B, rows<8> const &o, cplx<T> const *w, auto tw) {
            radix8<decltype(tw)::value, XS, YS, Last>(B, o, w, scale);
          });
      break;
    default:
      generic_stage<Last>(st, x, y, B, scale);
    }
  }

  /// Runs `butterfly(B, rows, twiddles, twiddle_flag)` for every (p, q).
  template <size_type R, typename PX, size_type XS, size_type YS,
            typename Butterfly>
  void stage_loop(stage const &st, layout<PX, XS> x, layout<T *, YS> y,
                  size_type B, Butterfly &&butterfly) const noexcept {
    size_type const m{st.m}, s{st.s};
    for (size_type p{0}; p < m; ++p) {
      cplx<T> const *w{_twiddles.data() + st.twiddle_offset + p * (R - 1)};
      for (size_type q{0}; q < s; ++q) {
        rows<R> io;
        for (size_type r{0}; r < R; ++r) {
          size_type const in{(q + s * (p + r * m)) * x.row};
          size_type const out{(q + s * (R * p + r)) * y.row};
          io.xr[r] = x.re + in;
          io.xi[r] = x.im + in;
          io.yr[r] = y.re + out;
          io.yi[r] = y.im + out;
        }
        if (p == 0)
          butterfly(B, io, w, std::false_type{});
        else
          butterfly(B, io, w, std::true_type{});
      }
    }
  }

  /// GCC applies restrict only to function parameters, so the row pointers
  /// are unpacked into the butterfly's parameter list.
  template <bool TW, size_type XS, size_type YS, bool Last>
  static void radix2(size_type B, rows<2> const &o, cplx<T> const *w,
                     T scale) noexcept {
    radix2<TW, XS, YS, Last>(B, o.xr[0], o.xi[0], o.xr[1], o.xi[1], o.yr[0],
                             o.yi[0], o.yr[1], o.yi[1], w, scale);
  }
  template <bool TW, size_type XS, size_type YS, bool Last>
  static void radix3(size_type B, rows<3> const &o, cplx<T> const *w,
                     T scale) noexcept {
    radix3<TW, XS, YS, Last>(B, o.xr[0], o.xi[0], o.xr[1], o.xi[1], o.xr[2],
                             o.xi[2], o.yr[0], o.yi[0], o.yr[1], o.yi[1],
                             o.yr[2], o.yi[2], w, scale);
  }
  template <bool TW, size_type XS, size_type YS, bool Last>
  static void radix4(size_type B, rows<4> const &o, cplx<T> const *w,
                     T scale) noexcept {
    radix4<TW, XS, YS, Last>(B, o.xr[0], o.xi[0], o.xr[1], o.xi[1], o.xr[2],
                             o.xi[2], o.xr[3], o.xi[3], o.yr[0], o.yi[0],
                             o.yr[1], o.yi[1], o.yr[2], o.yi[2], o.yr[3],
                             o.yi[3], w, scale);
  }
  template <bool TW, size_type XS, size_type YS, bool Last>
  static void radix5(size_type B, rows<5> const &o, cplx<T> const *w,
                     T scale) noexcept {
    radix5<TW, XS, YS, Last>(
        B, o.xr[0], o.xi[0], o.xr[1], o.xi[1], o.xr[2], o.xi[2], o.xr[3],
        o.xi[3], o.xr[4], o.xi[4], o.yr[0], o.yi[0], o.yr[1], o.yi[1], o.yr[2],
        o.yi[2], o.yr[3], o.yi[3], o.yr[4], o.yi[4], w, scale);
  }
  template <bool TW, size_type XS, size_type YS, bool Last>
  static void radix8(size_type B, rows<8> const &o, cplx<T> const *w,
                     T scale) noexcept {
    radix8<TW, XS, YS, Last>(
        B, o.xr[0], o.xi[0], o.xr[1], o.xi[1], o.xr[2], o.xi[2], o.xr[3],
        o.xi[3], o.xr[4], o.xi[4], o.xr[5], o.xi[5], o.xr[6], o.xi[6], o.xr[7],
        o.xi[7], o.yr[0], o.yi[0], o.yr[1], o.yi[1], o.yr[2], o.yi[2], o.yr[3],
        o.yi[3], o.yr[4], o.yi[4], o.yr[5], o.yi[5], o.yr[6], o.yi[6], o.yr[7],
        o.yi[7], w, scale);
  }

  /// Twiddle product (vr + i vi) * (wr + i wi) in place; identity when TW is
  /// false.
  template <bool TW>
  static constexpr void tw(T &vr, T &vi, T wr, T wi) noexcept {
    if constexpr (TW) {
      T const r{vr * wr - vi * wi};
      vi = vr * wi + vi * wr;
      vr = r;
    }
  }

  /// Output store; the scale is applied by the last stage only.
  template <size_type YS, bool Last>
  static constexpr void put(T *yr, T *yi, size_type b, T vr, T vi,
                            T scale) noexcept {
    if constexpr (Last) {
      yr[YS * b] = vr * scale;
      yi[YS * b] = vi * scale;
    } else {
      yr[YS * b] = vr;
      yi[YS * b] = vi;
    }
  }

  template <bool TW, size_type XS, size_type YS, bool Last>
  static void radix2(size_type B, T const *x0r, T const *x0i, T const *x1r,
                     T const *x1i, T *__restrict y0r, T *__restrict y0i,
                     T *__restrict y1r, T *__restrict y1i, cplx<T> const *w,
                     T scale) noexcept {
    T const w1r{w[0].real()}, w1i{w[0].imag()};
    for (size_type b{0}; b < B; ++b) {
      T const a0r{x0r[XS * b]}, a0i{x0i[XS * b]}, a1r{x1r[XS * b]},
          a1i{x1i[XS * b]};
      put<YS, Last>(y0r, y0i, b, a0r + a1r, a0i + a1i, scale);
      T v1r{a0r - a1r}, v1i{a0i - a1i};
      tw<TW>(v1r, v1i, w1r, w1i);
      put<YS, Last>(y1r, y1i, b, v1r, v1i, scale);
    }
  }

  template <bool TW, size_type XS, size_type YS, bool Last>
  static void radix3(size_type B, T const *x0r, T const *x0i, T const *x1r,
                     T const *x1i, T const *x2r, T const *x2i,
                     T *__restrict y0r, T *__restrict y0i, T *__restrict y1r,
                     T *__restrict y1i, T *__restrict y2r, T *__restrict y2i,
                     cplx<T> const *w, T scale) noexcept {
    constexpr T half{T(0.5)};
    constexpr T sin60{T(0.866025403784438646763723170752936183L)};
    T const w1r{w[0].real()}, w1i{w[0].imag()}, w2r{w[1].real()},
        w2i{w[1].imag()};
    for (size_type b{0}; b < B; ++b) {
      T const a0r{x0r[XS * b]}, a0i{x0i[XS * b]}, a1r{x1r[XS * b]},
          a1i{x1i[XS * b]};
      T const a2r{x2r[XS * b]}, a2i{x2i[XS * b]};
      T const t1r{a1r + a2r}, t1i{a1i + a2i};
      T const m1r{a0r - half * t1r}, m1i{a0i - half * t1i};
      // m2 = -i sin60 (a1 - a2)
      T const m2r{sin60 * (a1i - a2i)}, m2i{-sin60 * (a1r - a2r)};
      put<YS, Last>(y0r, y0i, b, a0r + t1r, a0i + t1i, scale);
      T v1r{m1r + m2r}, v1i{m1i + m2i}, v2r{m1r - m2r}, v2i{m1i - m2i};
      tw<TW>(v1r, v1i, w1r, w1i);
      tw<TW>(v2r, v2i, w2r, w2i);
      put<YS, Last>(y1r, y1i, b, v1r, v1i, scale);
      put<YS, Last>(y2r, y2i, b, v2r, v2i, scale);
    }
  }

  template <bool TW, size_type XS, size_type YS, bool Last>
  static void radix4(size_type B, T const *x0r, T const *x0i, T const *x1r,
                     T const *x1i, T const *x2r, T const *x2i, T const *x3r,
                     T const *x3i, T *__restrict y0r, T *__restrict y0i,
                     T *__restrict y1r, T *__restrict y1i, T *__restrict y2r,
                     T *__restrict y2i, T *__restrict y3r, T *__restrict y3i,
                     cplx<T> const *w, T scale) noexcept {
    T const w1r{w[0].real()}, w1i{w[0].imag()}, w2r{w[1].real()},
        w2i{w[1].imag()};
    T const w3r{w[2].real()}, w3i{w[2].imag()};
    for (size_type b{0}; b < B; ++b) {
      T const a0r{x0r[XS * b]}, a0i{x0i[XS * b]}, a1r{x1r[XS * b]},
          a1i{x1i[XS * b]};
      T const a2r{x2r[XS * b]}, a2i{x2i[XS * b]}, a3r{x3r[XS * b]},
          a3i{x3i[XS * b]};
      T const s02r{a0r + a2r}, s02i{a0i + a2i}, d02r{a0r - a2r},
          d02i{a0i - a2i};
      T const s13r{a1r + a3r}, s13i{a1i + a3i};
      // d13 = -i (a1 - a3)
      T const d13r{a1i - a3i}, d13i{a3r - a1r};
      put<YS, Last>(y0r, y0i, b, s02r + s13r, s02i + s13i, scale);
      T v1r{d02r + d13r}, v1i{d02i + d13i}, v2r{s02r - s13r}, v2i{s02i - s13i};
      T v3r{d02r - d13r}, v3i{d02i - d13i};
      tw<TW>(v1r, v1i, w1r, w1i);
      tw<TW>(v2r, v2i, w2r, w2i);
      tw<TW>(v3r, v3i, w3r, w3i);
      put<YS, Last>(y1r, y1i, b, v1r, v1i, scale);
      put<YS, Last>(y2r, y2i, b, v2r, v2i, scale);
      put<YS, Last>(y3r, y3i, b, v3r, v3i, scale);
    }
  }

  template <bool TW, size_type XS, size_type YS, bool Last>
  static void radix5(size_type B, T const *x0r, T const *x0i, T const *x1r,
                     T const *x1i, T const *x2r, T const *x2i, T const *x3r,
                     T const *x3i, T const *x4r, T const *x4i,
                     T *__restrict y0r, T *__restrict y0i, T *__restrict y1r,
                     T *__restrict y1i, T *__restrict y2r, T *__restrict y2i,
                     T *__restrict y3r, T *__restrict y3i, T *__restrict y4r,
                     T *__restrict y4i, cplx<T> const *w, T scale) noexcept {
    constexpr T c1{T(0.309016994374947424102293417182819059L)};  // cos(2pi/5)
    constexpr T c2{T(-0.809016994374947424102293417182819059L)}; // cos(4pi/5)
    constexpr T s1{T(0.951056516295153572116439333379382143L)};  // sin(2pi/5)
    constexpr T s2{T(0.587785252292473129168705954639072769L)};  // sin(4pi/5)
    T const w1r{w[0].real()}, w1i{w[0].imag()}, w2r{w[1].real()},
        w2i{w[1].imag()};
    T const w3r{w[2].real()}, w3i{w[2].imag()}, w4r{w[3].real()},
        w4i{w[3].imag()};
    for (size_type b{0}; b < B; ++b) {
      T const a0r{x0r[XS * b]}, a0i{x0i[XS * b]}, a1r{x1r[XS * b]},
          a1i{x1i[XS * b]};
      T const a2r{x2r[XS * b]}, a2i{x2i[XS * b]}, a3r{x3r[XS * b]},
          a3i{x3i[XS * b]};
      T const a4r{x4r[XS * b]}, a4i{x4i[XS * b]};
      T const t1r{a1r + a4r}, t1i{a1i + a4i}, t2r{a2r + a3r}, t2i{a2i + a3i};
      T const t3r{a1r - a4r}, t3i{a1i - a4i}, t4r{a2r - a3r}, t4i{a2i - a3i};
      T const p1r{a0r + c1 * t1r + c2 * t2r}, p1i{a0i + c1 * t1i + c2 * t2i};
      T const p2r{a0r + c2 * t1r + c1 * t2r}, p2i{a0i + c2 * t1i + c1 * t2i};
      // q1 = -i (s1 t3 + s2 t4), q2 = -i (s2 t3 - s1 t4)
      T const q1r{s1 * t3i + s2 * t4i}, q1i{-(s1 * t3r + s2 * t4r)};
      T const q2r{s2 * t3i - s1 * t4i}, q2i{-(s2 * t3r - s1 * t4r)};
      put<YS, Last>(y0r, y0i, b, a0r + t1r + t2r, a0i + t1i + t2i, scale);
      T v1r{p1r + q1r}, v1i{p1i + q1i}, v2r{p2r + q2r}, v2i{p2i + q2i};
      T v3r{p2r - q2r}, v3i{p2i - q2i}, v4r{p1r - q1r}, v4i{p1i - q1i};
      tw<TW>(v1r, v1i, w1r, w1i);
      tw<TW>(v2r, v2i, w2r, w2i);
      tw<TW>(v3r, v3i, w3r, w3i);
      tw<TW>(v4r, v4i, w4r, w4i);
      put<YS, Last>(y1r, y1i, b, v1r, v1i, scale);
      put<YS, Last>(y2r, y2i, b, v2r, v2i, scale);
      put<YS, Last>(y3r, y3i, b, v3r, v3i, scale);
      put<YS, Last>(y4r, y4i, b, v4r, v4i, scale);
    }
  }

  /// Radix-8 as two radix-4 DFTs (even / odd inputs) joined by radix-2
  /// butterflies, with the internal twiddles w_8^k folded into constants.
  template <bool TW, size_type XS, size_type YS, bool Last>
  static void radix8(size_type B, T const *x0r, T const *x0i, T const *x1r,
                     T const *x1i, T const *x2r, T const *x2i, T const *x3r,
                     T const *x3i, T const *x4r, T const *x4i, T const *x5r,
                     T const *x5i, T const *x6r, T const *x6i, T const *x7r,
                     T const *x7i, T *__restrict y0r, T *__restrict y0i,
                     T *__restrict y1r, T *__restrict y1i, T *__restrict y2r,
                     T *__restrict y2i, T *__restrict y3r, T *__restrict y3i,
                     T *__restrict y4r, T *__restrict y4i, T *__restrict y5r,
                     T *__restrict y5i, T *__restrict y6r, T *__restrict y6i,
                     T *__restrict y7r, T *__restrict y7i, cplx<T> const *w,
                     T scale) noexcept {
    constexpr T h{T(0.707106781186547524400844362104849039L)}; // 1/sqrt(2)
    T const w1r{w[0].real()}, w1i{w[0].imag()}, w2r{w[1].real()},
        w2i{w[1].imag()};
    T const w3r{w[2].real()}, w3i{w[2].imag()}, w4r{w[3].real()},
        w4i{w[3].imag()};
    T const w5r{w[4].real()}, w5i{w[4].imag()}, w6r{w[5].real()},
        w6i{w[5].imag()};
    T const w7r{w[6].real()}, w7i{w[6].imag()};
    for (size_type b{0}; b < B; ++b) {
      size_type const k{XS * b};
      // radix-4 DFT of the even inputs x0, x2, x4, x6
      T const e0r{x0r[k] + x4r[k]}, e0i{x0i[k] + x4i[k]}, e1r{x0r[k] - x4r[k]},
          e1i{x0i[k] - x4i[k]};
      T const e2r{x2r[k] + x6r[k]}, e2i{x2i[k] + x6i[k]}, e3r{x2r[k] - x6r[k]},
          e3i{x2i[k] - x6i[k]};
      T const E0r{e0r + e2r}, E0i{e0i + e2i}, E2r{e0r - e2r}, E2i{e0i - e2i};
      T const E1r{e1r + e3i}, E1i{e1i - e3r}; // e1 - i e3
      T const E3r{e1r - e3i}, E3i{e1i + e3r}; // e1 + i e3
      // radix-4 DFT of the odd inputs x1, x3, x5, x7
      T const o0r{x1r[k] + x5r[k]}, o0i{x1i[k] + x5i[k]}, o1r{x1r[k] - x5r[k]},
          o1i{x1i[k] - x5i[k]};
      T const o2r{x3r[k] + x7r[k]}, o2i{x3i[k] + x7i[k]}, o3r{x3r[k] - x7r[k]},
          o3i{x3i[k] - x7i[k]};
      T const O0r{o0r + o2r}, O0i{o0i + o2i}, O2r{o0r - o2r}, O2i{o0i - o2i};
      T const O1r{o1r + o3i}, O1i{o1i - o3r};
      T const O3r{o1r - o3i}, O3i{o1i + o3r};
      // odd outputs times w_8^k: k = 1: h (1 - i), k = 2: -i, k = 3: -h (1 + i)
      T const t1r{h * (O1r + O1i)}, t1i{h * (O1i - O1r)};
      T const t2r{O2i}, t2i{-O2r};
      T const t3r{h * (O3i - O3r)}, t3i{-h * (O3r + O3i)};
      put<YS, Last>(y0r, y0i, b, E0r + O0r, E0i + O0i, scale);
      T v1r{E1r + t1r}, v1i{E1i + t1i}, v2r{E2r + t2r}, v2i{E2i + t2i};
      T v3r{E3r + t3r}, v3i{E3i + t3i}, v4r{E0r - O0r}, v4i{E0i - O0i};
      T v5r{E1r - t1r}, v5i{E1i - t1i}, v6r{E2r - t2r}, v6i{E2i - t2i};
      T v7r{E3r - t3r}, v7i{E3i - t3i};
      tw<TW>(v1r, v1i, w1r, w1i);
      tw<TW>(v2r, v2i, w2r, w2i);
      tw<TW>(v3r, v3i, w3r, w3i);
      tw<TW>(v4r, v4i, w4r, w4i);
      tw<TW>(v5r, v5i, w5r, w5i);
      tw<TW>(v6r, v6i, w6r, w6i);
      tw<TW>(v7r, v7i, w7r, w7i);
      put<YS, Last>(y1r, y1i, b, v1r, v1i, scale);
      put<YS, Last>(y2r, y2i, b, v2r, v2i, scale);
      put<YS, Last>(y3r, y3i, b, v3r, v3i, scale);
      put<YS, Last>(y4r, y4i, b, v4r, v4i, scale);
      put<YS, Last>(y5r, y5i, b, v5r, v5i, scale);
      put<YS, Last>(y6r, y6i, b, v6r, v6i, scale);
      put<YS, Last>(y7r, y7i, b, v7r, v7i, scale);
    }
  }

  /// Direct O(R^2) DFT butterfly for odd primes 7 .. max_direct_radix,
  /// with the batch as the innermost loop.
  template <bool Last, typename PX, size_type XS, size_type YS>
  void generic_stage(stage const &st, layout<PX, XS> x, layout<T *, YS> y,
                     size_type B, T scale) const noexcept {
    size_type const R{st.radix}, m{st.m}, s{st.s};
    cplx<T> const *roots{_roots.data() + st.roots_offset};
    for (size_type p{0}; p < m; ++p) {
      cplx<T> const *w{_twiddles.data() + st.twiddle_offset + p * (R - 1)};
      for (size_type q{0}; q < s; ++q) {
        for (size_type t{0}; t < R; ++t) {
          T *__restrict yr{y.re + (q + s * (R * p + t)) * y.row};
          T *__restrict yi{y.im + (q + s * (R * p + t)) * y.row};
          T const *__restrict x0r{x.re + (q + s * p) * x.row};
          T const *__restrict x0i{x.im + (q + s * p) * x.row};
          for (size_type b{0}; b < B; ++b) {
            yr[YS * b] = x0r[XS * b];
            yi[YS * b] = x0i[XS * b];
          }
          size_type k{0};
          for (size_type r{1}; r < R; ++r) {
            k += t;
            if (k >= R)
              k -= R;
            T const cr{roots[k].real()}, ci{roots[k].imag()};
            T const *__restrict xr{x.re + (q + s * (p + r * m)) * x.row};
            T const *__restrict xi{x.im + (q + s * (p + r * m)) * x.row};
            for (size_type b{0}; b < B; ++b) {
              yr[YS * b] += xr[XS * b] * cr - xi[XS * b] * ci;
              yi[YS * b] += xr[XS * b] * ci + xi[XS * b] * cr;
            }
          }
          if (t > 0 && p > 0) {
            T const wr{w[t - 1].real()}, wi{w[t - 1].imag()};
            for (size_type b{0}; b < B; ++b) {
              T const vr{yr[YS * b]}, vi{yi[YS * b]};
              yr[YS * b] = vr * wr - vi * wi;
              yi[YS * b] = vr * wi + vi * wr;
            }
          }
          if constexpr (Last) {
            for (size_type b{0}; b < B; ++b) {
              yr[YS * b] *= scale;
              yi[YS * b] *= scale;
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

} // namespace numsim::fft::kernel

#endif // NUMSIM_FFT_KERNEL_STOCKHAM_H
