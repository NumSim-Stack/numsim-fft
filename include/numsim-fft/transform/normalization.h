#ifndef NUMSIM_FFT_TRANSFORM_NORMALIZATION_H
#define NUMSIM_FFT_TRANSFORM_NORMALIZATION_H

namespace numsim_fft {

/**
 * @brief Where the factor 1/N goes, with N the product over the axes of
 * logical_size(kind, n) (N = number of points for periodic axes).
 *  - none:     neither direction is scaled; backward(forward(x)) = N x.
 *  - backward: backward is scaled by 1/N (numpy's default).
 *  - ortho:    both directions are scaled by 1/sqrt(N).
 */
enum class normalization { none, backward, ortho };

} // namespace numsim_fft

#endif // NUMSIM_FFT_TRANSFORM_NORMALIZATION_H
