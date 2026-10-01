#ifndef NUMSIM_FFT_FIELD_FIELD_H
#define NUMSIM_FFT_FIELD_FIELD_H

#include "../core/aligned_allocator.h"
#include "../core/mdspan.h"
#include "../layout/extents.h"
#include "element_traits.h"

#include <array>
#include <concepts>
#include <cstddef>
#include <memory>
#include <ranges>
#include <span>
#include <stdexcept>
#include <vector>

namespace numsim::fft {

/**
 * @brief Values of type Element on a SpatialDim-dimensional regular grid.
 *
 * Storage is one contiguous, aligned scalar buffer in AoS order
 * `[i0]...[i_{d-1}][component]` (row-major points, the components of a point
 * adjacent). Points are accessed through element_traits<Element>::reference,
 * a tmech view for tensor elements, so pointwise tensor algebra works in
 * place:
 *
 * @code
 * field<tmech::tensor<double,3,2>, 3> eps{extents{16, 16, 16}};
 * eps[i, j, k] = tmech::sym(grad);
 * sig[i, j, k] = tmech::dcontract(C, eps[i, j, k]);
 * @endcode
 *
 * The point accessors return *views* (tensor_ref), not copies:
 *  - `auto v = eps[i, j, k];` refers to the field's memory and writes
 *    through; it dangles once the field is destroyed or reallocated.
 *  - `tensor2 t{eps[i, j, k]};` or to_vector() make copies.
 *  - assigning an expression that reads the same point
 *    (`eps[p] = tmech::trans(eps[p])`) aliases, as with tmech::tensor; wrap
 *    it in tmech::eval().
 */
template <field_element Element, std::size_t SpatialDim,
          typename Allocator = aligned_allocator<typename element_traits<Element>::scalar_type>>
class field {
public:
  using traits = element_traits<Element>;
  using element_type = Element;
  using scalar_type = typename traits::scalar_type;
  using allocator_type = Allocator;
  using extents_type = numsim::fft::extents<SpatialDim>;
  using index_type = typename extents_type::index_type;
  using reference = typename traits::reference;
  using const_reference = typename traits::const_reference;
  using size_type = std::size_t;

  static constexpr size_type components{traits::components};
  static constexpr size_type spatial_dimension{SpatialDim};

  /// Same element and grid with another scalar type (e.g. the spectrum).
  template <scalar U>
  using rebind_scalar =
      field<typename traits::template rebind<U>, SpatialDim,
            typename std::allocator_traits<Allocator>::template rebind_alloc<U>>;

  field() = default;

  /// Zero-initialised field.
  explicit field(extents_type const &e, Allocator const &alloc = Allocator{})
      : _extents{e}, _data(e.size() * components, scalar_type{}, alloc) {}

  /// Field with every point set to `value`.
  field(extents_type const &e, Element const &value, Allocator const &alloc = Allocator{})
      : field(e, alloc) {
    fill(value);
  }

  extents_type const &extents() const noexcept { return _extents; }

  /// Number of grid points.
  size_type size() const noexcept { return _extents.size(); }

  /// Number of scalars (points * components).
  size_type scalar_size() const noexcept { return _data.size(); }

  scalar_type *data() noexcept { return _data.data(); }
  scalar_type const *data() const noexcept { return _data.data(); }

  std::span<scalar_type> scalars() noexcept { return _data; }
  std::span<scalar_type const> scalars() const noexcept { return _data; }

  /// Point by linear (row-major) index.
  reference point(size_type linear) noexcept {
    return traits::make_reference(_data.data() + linear * components);
  }
  const_reference point(size_type linear) const noexcept {
    return traits::make_reference(_data.data() + linear * components);
  }

  template <std::integral... I>
    requires(sizeof...(I) == SpatialDim)
  reference operator[](I... idx) noexcept {
    return point(_extents.linear_index(index_type{static_cast<size_type>(idx)...}));
  }
  template <std::integral... I>
    requires(sizeof...(I) == SpatialDim)
  const_reference operator[](I... idx) const noexcept {
    return point(_extents.linear_index(index_type{static_cast<size_type>(idx)...}));
  }

  reference operator[](index_type const &idx) noexcept {
    return point(_extents.linear_index(idx));
  }
  const_reference operator[](index_type const &idx) const noexcept {
    return point(_extents.linear_index(idx));
  }

  void fill(Element const &value) {
    for (size_type p{0}; p < size(); ++p)
      traits::store(value, _data.data() + p * components);
  }

  /// Copies size() values (row-major point order) into the field.
  template <std::ranges::forward_range R>
    requires std::convertible_to<std::ranges::range_reference_t<R>, Element const &>
  void assign(R const &values) {
    if (static_cast<size_type>(std::ranges::distance(values)) != size())
      throw std::invalid_argument("field::assign: size mismatch");
    size_type p{0};
    for (Element const &v : values)
      traits::store(v, _data.data() + (p++) * components);
  }

  /// Copies the points out (row-major).
  std::vector<Element> to_vector() const {
    std::vector<Element> out;
    out.reserve(size());
    for (size_type p{0}; p < size(); ++p)
      out.push_back(traits::load(_data.data() + p * components));
    return out;
  }

  /// Scalar mdspan over the buffer, extents (n_0, ..., n_{d-1}, components).
  auto mdspan() noexcept { return make_mdspan(_data.data()); }
  auto mdspan() const noexcept { return make_mdspan(_data.data()); }

  allocator_type get_allocator() const noexcept { return _data.get_allocator(); }

private:
  template <typename S> auto make_mdspan(S *ptr) const noexcept {
    std::array<size_type, SpatialDim + 1> ext{};
    for (size_type d{0}; d < SpatialDim; ++d)
      ext[d] = _extents[d];
    ext[SpatialDim] = components;
    return md::mdspan<S, md::dextents<size_type, SpatialDim + 1>>{ptr, ext};
  }

  extents_type _extents{};
  std::vector<scalar_type, Allocator> _data;
};

} // namespace numsim::fft

#endif // NUMSIM_FFT_FIELD_FIELD_H
