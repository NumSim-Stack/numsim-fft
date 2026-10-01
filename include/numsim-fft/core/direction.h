#ifndef NUMSIM_FFT_CORE_DIRECTION_H
#define NUMSIM_FFT_CORE_DIRECTION_H

namespace numsim_fft {

/// Sign of the exponent: forward exp(-2 pi i jk/N), backward exp(+2 pi i jk/N).
enum class direction : int { forward = -1, backward = +1 };

constexpr direction reverse(direction d) noexcept {
  return d == direction::forward ? direction::backward : direction::forward;
}

} // namespace numsim_fft

#endif // NUMSIM_FFT_CORE_DIRECTION_H
