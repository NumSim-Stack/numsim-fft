#ifndef NUMSIM_FFT_FIELD_ALGEBRA_H
#define NUMSIM_FFT_FIELD_ALGEBRA_H

#include "../core/scalar_traits.h"
#include "../execution/executor.h"
#include "../execution/sequential.h"
#include "field.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <vector>

namespace numsim::fft {

/**
 * @file Vector-space operations on fields (the building blocks of
 * matrix-free Krylov solvers), plus a parallel loop over points.
 *
 * Reductions split the scalars into a fixed number of chunks and sum the
 * chunk results in order, so the result does not depend on the executor or
 * the number of threads.
 */

namespace detail {

inline constexpr std::size_t reduction_chunks{256};

/// Calls f(first, last) for each chunk of [0, n) and sums the results in
/// chunk order.
template <typename R, typename Exec, typename F>
R chunked_sum(std::size_t n, Exec const &exec, F const &f) {
  if (n == 0)
    return R{};
  std::size_t const chunks{std::min<std::size_t>(reduction_chunks, n)};
  std::vector<R> partial(chunks);
  exec.bulk(chunks, [&](std::size_t c) { partial[c] = f(c * n / chunks, (c + 1) * n / chunks); });
  R sum{};
  for (auto const &p : partial)
    sum += p;
  return sum;
}

/// Sum of f(i) over [i0, i1) with `lanes` independent accumulators, so the
/// loop vectorises without reassociation flags; the summation order is
/// fixed, hence the result is reproducible.
template <typename R, std::size_t lanes = 8, typename F>
R lane_sum(std::size_t i0, std::size_t i1, F const &f) {
  R acc[lanes]{};
  std::size_t i{i0};
  for (; i + lanes <= i1; i += lanes)
    for (std::size_t l{0}; l < lanes; ++l)
      acc[l] += f(i + l);
  for (std::size_t l{0}; i < i1; ++i, ++l)
    acc[l] += f(i);
  R sum{};
  for (std::size_t l{0}; l < lanes; ++l)
    sum += acc[l];
  return sum;
}

template <typename Exec, typename F> void chunked_for(std::size_t n, Exec const &exec, F const &f) {
  if (n == 0)
    return;
  std::size_t const workers{std::max<std::size_t>(1, exec.concurrency())};
  std::size_t const chunks{std::min<std::size_t>(n, workers == 1 ? 1 : 8 * workers)};
  exec.bulk(chunks, [&](std::size_t c) { f(c * n / chunks, (c + 1) * n / chunks); });
}

} // namespace detail

/// Inner product over all scalars: sum a_i b_i (real) or sum conj(a_i) b_i.
template <typename E, std::size_t D, typename A, executor Exec = sequential_executor>
auto dot(field<E, D, A> const &a, field<E, D, A> const &b, Exec const &exec = {}) {
  using S = typename field<E, D, A>::scalar_type;
  return detail::chunked_sum<S>(a.scalar_size(), exec, [&](std::size_t i0, std::size_t i1) {
    S const *__restrict pa{a.data()};
    S const *__restrict pb{b.data()};
    return detail::lane_sum<S>(i0, i1, [&](std::size_t i) {
      if constexpr (is_complex_v<S>)
        return std::conj(pa[i]) * pb[i];
      else
        return pa[i] * pb[i];
    });
  });
}

/// Squared Euclidean norm over all scalars (real for complex fields too).
template <typename E, std::size_t D, typename A, executor Exec = sequential_executor>
auto norm2(field<E, D, A> const &a, Exec const &exec = {}) {
  using S = typename field<E, D, A>::scalar_type;
  using R = real_type_t<S>;
  return detail::chunked_sum<R>(a.scalar_size(), exec, [&](std::size_t i0, std::size_t i1) {
    S const *__restrict pa{a.data()};
    return detail::lane_sum<R>(i0, i1,
                               [&](std::size_t i) { return static_cast<R>(std::norm(pa[i])); });
  });
}

template <typename E, std::size_t D, typename A, executor Exec = sequential_executor>
auto norm(field<E, D, A> const &a, Exec const &exec = {}) {
  return std::sqrt(norm2(a, exec));
}

/// y += alpha * x
template <typename E, std::size_t D, typename A, typename Scalar,
          executor Exec = sequential_executor>
void axpy(Scalar alpha, field<E, D, A> const &x, field<E, D, A> &y, Exec const &exec = {}) {
  detail::chunked_for(x.scalar_size(), exec, [&](std::size_t i0, std::size_t i1) {
    auto const *px{x.data()};
    auto *py{y.data()};
    for (std::size_t i{i0}; i < i1; ++i)
      py[i] += alpha * px[i];
  });
}

/// x *= alpha
template <typename E, std::size_t D, typename A, typename Scalar,
          executor Exec = sequential_executor>
void scale(Scalar alpha, field<E, D, A> &x, Exec const &exec = {}) {
  detail::chunked_for(x.scalar_size(), exec, [&](std::size_t i0, std::size_t i1) {
    auto *px{x.data()};
    for (std::size_t i{i0}; i < i1; ++i)
      px[i] *= alpha;
  });
}

/// Sets every scalar to `value`.
template <typename E, std::size_t D, typename A, executor Exec = sequential_executor>
void fill(field<E, D, A> &x, typename field<E, D, A>::scalar_type value, Exec const &exec = {}) {
  detail::chunked_for(x.scalar_size(), exec, [&](std::size_t i0, std::size_t i1) {
    std::fill(x.data() + i0, x.data() + i1, value);
  });
}

/// Calls f(point_index, view) for every point, possibly in parallel.
template <typename E, std::size_t D, typename A, typename F, executor Exec = sequential_executor>
void for_each_point(field<E, D, A> &x, F const &f, Exec const &exec = {}) {
  detail::chunked_for(x.size(), exec, [&](std::size_t p0, std::size_t p1) {
    for (std::size_t p{p0}; p < p1; ++p)
      f(p, x.point(p));
  });
}

template <typename E, std::size_t D, typename A, typename F, executor Exec = sequential_executor>
void for_each_point(field<E, D, A> const &x, F const &f, Exec const &exec = {}) {
  detail::chunked_for(x.size(), exec, [&](std::size_t p0, std::size_t p1) {
    for (std::size_t p{p0}; p < p1; ++p)
      f(p, x.point(p));
  });
}

} // namespace numsim::fft

#endif // NUMSIM_FFT_FIELD_ALGEBRA_H
