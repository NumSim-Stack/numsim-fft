#ifndef NUMSIM_FFT_TRANSFORM_API_H
#define NUMSIM_FFT_TRANSFORM_API_H

// One-shot transforms: build a plan, run it, return the new field. For
// repeated transforms on the same grid, create a plan once (plan.h).

#include "plan.h"

#include <complex>

namespace numsim_fft {

namespace detail {
template <typename F> using real_of_t = real_type_t<typename F::scalar_type>;
} // namespace detail

/// Complex forward transform (periodic axes by default).
template <typename E, std::size_t D, typename A, executor Exec = sequential_executor>
  requires complex_scalar<typename element_traits<E>::scalar_type>
[[nodiscard]] expected<field<E, D, A>, error> fft(field<E, D, A> const &x,
                                         std::array<axis_kind, D> const &kinds = {},
                                         plan_options const &options = {}, Exec const &exec = {}) {
  using T = detail::real_of_t<field<E, D, A>>;
  return make_c2c_plan<T>(x.extents(), kinds, options)
      .and_then([&](plan<T, D> const &p) -> expected<field<E, D, A>, error> {
        field<E, D, A> out{p.spectral_extents()};
        return p.forward(x, out, exec).transform([&] { return std::move(out); });
      });
}

/// Complex backward transform, the inverse of fft() (normalised by default).
template <typename E, std::size_t D, typename A, executor Exec = sequential_executor>
  requires complex_scalar<typename element_traits<E>::scalar_type>
[[nodiscard]] expected<field<E, D, A>, error> ifft(field<E, D, A> const &x,
                                          std::array<axis_kind, D> const &kinds = {},
                                          plan_options const &options = {}, Exec const &exec = {}) {
  using T = detail::real_of_t<field<E, D, A>>;
  return make_c2c_plan<T>(x.extents(), kinds, options)
      .and_then([&](plan<T, D> const &p) -> expected<field<E, D, A>, error> {
        field<E, D, A> out{p.physical_extents()};
        return p.backward(x, out, exec).transform([&] { return std::move(out); });
      });
}

/// Real-to-complex forward transform; returns the half spectrum.
template <typename E, std::size_t D, typename A, executor Exec = sequential_executor>
  requires real_scalar<typename element_traits<E>::scalar_type>
[[nodiscard]] auto rfft(field<E, D, A> const &x, std::array<axis_kind, D> const &kinds = {},
          plan_options const &options = {}, Exec const &exec = {}) {
  using T = detail::real_of_t<field<E, D, A>>;
  using out_field = typename field<E, D, A>::template rebind_scalar<std::complex<T>>;
  return make_r2c_plan<T>(x.extents(), kinds, options)
      .and_then([&](plan<T, D> const &p) -> expected<out_field, error> {
        out_field out{p.spectral_extents()};
        return p.forward(x, out, exec).transform([&] { return std::move(out); });
      });
}

/// Inverse of rfft(): `physical` is the grid of the real field.
template <typename E, std::size_t D, typename A, executor Exec = sequential_executor>
  requires complex_scalar<typename element_traits<E>::scalar_type>
[[nodiscard]] auto irfft(field<E, D, A> const &X, extents<D> const &physical,
           std::array<axis_kind, D> const &kinds = {}, plan_options const &options = {},
           Exec const &exec = {}) {
  using T = detail::real_of_t<field<E, D, A>>;
  using out_field = typename field<E, D, A>::template rebind_scalar<T>;
  return make_r2c_plan<T>(physical, kinds, options)
      .and_then([&](plan<T, D> const &p) -> expected<out_field, error> {
        out_field out{p.physical_extents()};
        return p.backward(X, out, exec).transform([&] { return std::move(out); });
      });
}

/// Real-to-real (DCT/DST) forward transform.
template <typename E, std::size_t D, typename A, executor Exec = sequential_executor>
  requires real_scalar<typename element_traits<E>::scalar_type>
[[nodiscard]] expected<field<E, D, A>, error> r2r(field<E, D, A> const &x,
                                         std::array<axis_kind, D> const &kinds,
                                         plan_options const &options = {}, Exec const &exec = {}) {
  using T = detail::real_of_t<field<E, D, A>>;
  return make_r2r_plan<T>(x.extents(), kinds, options)
      .and_then([&](plan<T, D> const &p) -> expected<field<E, D, A>, error> {
        field<E, D, A> out{p.spectral_extents()};
        return p.forward(x, out, exec).transform([&] { return std::move(out); });
      });
}

/// Inverse of r2r() with the same `kinds` (applies the inverse kinds).
template <typename E, std::size_t D, typename A, executor Exec = sequential_executor>
  requires real_scalar<typename element_traits<E>::scalar_type>
[[nodiscard]] expected<field<E, D, A>, error> inverse_r2r(field<E, D, A> const &X,
                                                 std::array<axis_kind, D> const &kinds,
                                                 plan_options const &options = {},
                                                 Exec const &exec = {}) {
  using T = detail::real_of_t<field<E, D, A>>;
  return make_r2r_plan<T>(X.extents(), kinds, options)
      .and_then([&](plan<T, D> const &p) -> expected<field<E, D, A>, error> {
        field<E, D, A> out{p.physical_extents()};
        return p.backward(X, out, exec).transform([&] { return std::move(out); });
      });
}

} // namespace numsim_fft

#endif // NUMSIM_FFT_TRANSFORM_API_H
