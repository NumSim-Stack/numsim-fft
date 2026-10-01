#ifndef NUMSIM_FFT_LAYOUT_EXTENTS_H
#define NUMSIM_FFT_LAYOUT_EXTENTS_H

#include <array>
#include <concepts>
#include <cstddef>

namespace numsim_fft {

/**
 * @brief Number of grid points per axis of a row-major (last axis fastest)
 * grid with Dim spatial dimensions.
 */
template <std::size_t Dim> class extents {
  static_assert(Dim >= 1, "extents: at least one dimension");

public:
  using size_type = std::size_t;
  using index_type = std::array<size_type, Dim>;

  constexpr extents() noexcept = default;

  template <std::integral... I>
    requires(sizeof...(I) == Dim)
  constexpr explicit(Dim == 1) extents(I... n) noexcept
      : _n{static_cast<size_type>(n)...} {}

  constexpr explicit extents(index_type const &n) noexcept : _n{n} {}

  static constexpr size_type rank() noexcept { return Dim; }

  constexpr size_type operator[](size_type axis) const noexcept { return _n[axis]; }

  constexpr index_type const &as_array() const noexcept { return _n; }

  /// Total number of grid points.
  constexpr size_type size() const noexcept {
    size_type s{1};
    for (auto n : _n)
      s *= n;
    return s;
  }

  /// Product of the extents before `axis`.
  constexpr size_type outer(size_type axis) const noexcept {
    size_type s{1};
    for (size_type d{0}; d < axis; ++d)
      s *= _n[d];
    return s;
  }

  /// Product of the extents after `axis`, i.e. the stride of `axis` in points.
  constexpr size_type inner(size_type axis) const noexcept {
    size_type s{1};
    for (size_type d{axis + 1}; d < Dim; ++d)
      s *= _n[d];
    return s;
  }

  constexpr size_type stride(size_type axis) const noexcept { return inner(axis); }

  constexpr size_type linear_index(index_type const &idx) const noexcept {
    size_type l{0};
    for (size_type d{0}; d < Dim; ++d)
      l = l * _n[d] + idx[d];
    return l;
  }

  constexpr index_type multi_index(size_type linear) const noexcept {
    index_type idx{};
    for (size_type d{Dim}; d-- > 0;) {
      idx[d] = linear % _n[d];
      linear /= _n[d];
    }
    return idx;
  }

  /// Copy with the extent of `axis` replaced by `n`.
  constexpr extents with_extent(size_type axis, size_type n) const noexcept {
    extents e{*this};
    e._n[axis] = n;
    return e;
  }

  friend constexpr bool operator==(extents const &, extents const &) noexcept = default;

private:
  index_type _n{};
};

template <std::integral... I> extents(I...) -> extents<sizeof...(I)>;

} // namespace numsim_fft

#endif // NUMSIM_FFT_LAYOUT_EXTENTS_H
