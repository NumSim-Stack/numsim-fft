#ifndef NUMSIM_FFT_CORE_SCALAR_TRAITS_H
#define NUMSIM_FFT_CORE_SCALAR_TRAITS_H

#include <complex>
#include <concepts>
#include <numbers>
#include <type_traits>

namespace numsim_fft {

template <typename T> struct is_complex : std::false_type {};
template <typename T> struct is_complex<std::complex<T>> : std::true_type {};

/// True for std::complex<T> (cv-qualifiers ignored).
template <typename T>
inline constexpr bool is_complex_v = is_complex<std::remove_cv_t<T>>::value;

/**
 * @brief Real scalar types the transforms are defined for.
 *
 * Restricted to the standard floating-point types, which is what std::complex
 * is specified for and what the tests cover.
 */
template <typename T>
concept real_scalar = std::floating_point<T>;

/// std::complex over a real_scalar.
template <typename T>
concept complex_scalar =
    is_complex_v<T> && real_scalar<typename std::remove_cv_t<T>::value_type>;

/// Scalar value type of a field: real or complex.
template <typename T>
concept scalar = real_scalar<T> || complex_scalar<T>;

namespace detail {
template <typename T> struct real_type {
  using type = std::remove_cv_t<T>;
};
template <typename T> struct real_type<std::complex<T>> {
  using type = T;
};
} // namespace detail

/// Underlying real type: T for real T, T for std::complex<T>.
template <typename T>
using real_type_t = typename detail::real_type<std::remove_cv_t<T>>::type;

/// std::complex<real_type_t<T>>.
template <typename T> using complex_type_t = std::complex<real_type_t<T>>;

/// pi in the precision of T.
template <real_scalar T> inline constexpr T pi_v = std::numbers::pi_v<T>;

} // namespace numsim_fft

#endif // NUMSIM_FFT_CORE_SCALAR_TRAITS_H
