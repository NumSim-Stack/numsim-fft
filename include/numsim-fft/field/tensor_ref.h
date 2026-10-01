#ifndef NUMSIM_FFT_FIELD_TENSOR_REF_H
#define NUMSIM_FFT_FIELD_TENSOR_REF_H

#include <tmech/tmech.h>

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <type_traits>

namespace numsim_fft {

/**
 * @brief Non-owning view of one tensor inside a field buffer.
 *
 * It is a tmech::adaptor (full storage, row-major like tmech::tensor), so it
 * takes part in every tmech expression. On top of the adaptor it provides
 *  - value semantics for copy assignment: `f[i] = f[j]` copies the tensor.
 *    The adaptor's implicit copy assignment would rebind the pointer.
 *  - mutable component access: `f[i](0, 1) = x`.
 *
 * T may be const for read-only views.
 *
 * As with any tmech target, assigning an expression that reads the viewed
 * tensor itself (e.g. `v = trans(v)`) aliases; use tmech::eval() there.
 */
template <typename T, std::size_t Dim, std::size_t Rank>
class tensor_ref : public tmech::adaptor<T, Dim, Rank, tmech::full<Dim>> {
  using base = tmech::adaptor<T, Dim, Rank, tmech::full<Dim>>;

  static constexpr std::size_t ipow(std::size_t b, std::size_t e) noexcept {
    std::size_t r{1};
    while (e-- > 0)
      r *= b;
    return r;
  }

public:
  using value_type = std::remove_const_t<T>;
  using size_type = std::size_t;

  static constexpr size_type size() noexcept { return ipow(Dim, Rank); }

  constexpr explicit tensor_ref(T *data) noexcept : base{data}, _ptr{data} {}

  constexpr tensor_ref(tensor_ref const &other) noexcept = default;

  /// Mutable view from a mutable one of the same element type.
  constexpr operator tensor_ref<T const, Dim, Rank>() const noexcept
    requires(!std::is_const_v<T>)
  {
    return tensor_ref<T const, Dim, Rank>{_ptr};
  }

  /// Copies the values of `other` (never rebinds).
  constexpr tensor_ref &operator=(tensor_ref const &other) noexcept
    requires(!std::is_const_v<T>)
  {
    std::copy_n(other._ptr, size(), _ptr);
    return *this;
  }

  /// Evaluates a tmech expression (or tensor, or other view) into the view.
  template <typename Derived>
  constexpr tensor_ref &operator=(tmech::tensor_base<Derived> const &expr) noexcept
    requires(!std::is_const_v<T>)
  {
    base::operator=(expr);
    return *this;
  }

  /// Component access; mutable for non-const T.
  template <std::integral... I>
    requires(sizeof...(I) == Rank)
  constexpr T &operator()(I... idx) const noexcept {
    size_type flat{0};
    ((flat = flat * Dim + static_cast<size_type>(idx)), ...);
    return _ptr[flat];
  }

  constexpr T *data() const noexcept { return _ptr; }

private:
  T *_ptr;
};

} // namespace numsim_fft

namespace tmech {
/// tmech accepts operands through this trait (one specialisation per type);
/// tensor_ref provides the adaptor interface, so it qualifies.
template <typename T, std::size_t Dim, std::size_t Rank>
struct is_tensor_type<numsim_fft::tensor_ref<T, Dim, Rank>> {
  using type = std::true_type;
  static constexpr bool value = true;
};
} // namespace tmech

#endif // NUMSIM_FFT_FIELD_TENSOR_REF_H
