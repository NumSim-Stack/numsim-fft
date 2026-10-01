#ifndef NUMSIM_FFT_DISTRIBUTED_SLAB_DECOMPOSITION_H
#define NUMSIM_FFT_DISTRIBUTED_SLAB_DECOMPOSITION_H

#include <cstddef>

namespace numsim::fft {

/**
 * @brief Block distribution of `global` indices over `parts` parts.
 *
 * Part p holds global/parts indices, plus one more if p < global % parts,
 * starting at offset(p). Parts may be empty when parts > global.
 */
class slab_decomposition {
public:
  using size_type = std::size_t;

  constexpr slab_decomposition(size_type global, size_type parts) noexcept
      : _global{global}, _parts{parts} {}

  constexpr size_type global_size() const noexcept { return _global; }
  constexpr size_type parts() const noexcept { return _parts; }

  constexpr size_type local_size(size_type part) const noexcept {
    return _global / _parts + (part < _global % _parts ? 1 : 0);
  }

  constexpr size_type offset(size_type part) const noexcept {
    size_type const base{_global / _parts}, extra{_global % _parts};
    return part * base + (part < extra ? part : extra);
  }

private:
  size_type _global;
  size_type _parts;
};

} // namespace numsim::fft

#endif // NUMSIM_FFT_DISTRIBUTED_SLAB_DECOMPOSITION_H
