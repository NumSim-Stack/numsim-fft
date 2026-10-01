#ifndef NUMSIM_FFT_FIELD_ELEMENT_TRAITS_H
#define NUMSIM_FFT_FIELD_ELEMENT_TRAITS_H

#include "../core/scalar_traits.h"
#include "tensor_ref.h"

#include <tmech/tmech.h>

#include <algorithm>
#include <cstddef>

namespace numsim_fft {

/**
 * @brief Describes how one grid-point value is stored in a field buffer.
 *
 * Members of a specialisation:
 *  - scalar_type: real or complex scalar of the storage.
 *  - components: number of scalars per point.
 *  - reference / const_reference and make_reference(ptr): view of a point.
 *  - store(value, ptr) / load(ptr): copy a value in / out.
 *  - rebind<U>: the same element with scalar type U (e.g. real -> complex).
 *
 * The primary template is empty; types without a specialisation are not
 * field elements.
 */
template <typename E> struct element_traits {};

/// Scalars: one component, plain references.
template <scalar T> struct element_traits<T> {
  using element_type = T;
  using scalar_type = T;
  using reference = T &;
  using const_reference = T const &;

  static constexpr std::size_t components{1};

  static constexpr reference make_reference(T *p) noexcept { return *p; }
  static constexpr const_reference make_reference(T const *p) noexcept { return *p; }
  static constexpr void store(T const &value, T *p) noexcept { *p = value; }
  static constexpr T load(T const *p) noexcept { return *p; }

  template <scalar U> using rebind = U;
};

/**
 * @brief tmech tensors: Dim^Rank components in tmech's row-major order,
 * viewed through numsim_fft::tensor_ref. Ranks 1, 2 and 4 (the ranks
 * tmech::full storage supports).
 */
template <scalar T, std::size_t Dim, std::size_t Rank>
  requires(Rank == 1 || Rank == 2 || Rank == 4)
struct element_traits<tmech::tensor<T, Dim, Rank>> {
  using element_type = tmech::tensor<T, Dim, Rank>;
  using scalar_type = T;
  using reference = tensor_ref<T, Dim, Rank>;
  using const_reference = tensor_ref<T const, Dim, Rank>;

  static constexpr std::size_t components{reference::size()};

  static constexpr reference make_reference(T *p) noexcept { return reference{p}; }
  static constexpr const_reference make_reference(T const *p) noexcept {
    return const_reference{p};
  }
  static constexpr void store(element_type const &value, T *p) noexcept {
    std::copy_n(value.raw_data(), components, p);
  }
  static constexpr element_type load(T const *p) noexcept {
    element_type t;
    std::copy_n(p, components, t.raw_data());
    return t;
  }

  template <scalar U> using rebind = tmech::tensor<U, Dim, Rank>;
};

/// Types a field can hold.
template <typename E>
concept field_element = requires {
  typename element_traits<E>::scalar_type;
  { element_traits<E>::components } -> std::convertible_to<std::size_t>;
};

} // namespace numsim_fft

#endif // NUMSIM_FFT_FIELD_ELEMENT_TRAITS_H
