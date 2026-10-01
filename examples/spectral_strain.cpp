// Small-strain tensor eps = sym(grad u) of a periodic displacement field,
// computed spectrally: eps_hat_ij = i/2 (k_j u_hat_i + k_i u_hat_j).
//
// u is a field of tmech rank-1 tensors, eps a field of rank-2 tensors.

#include <numsim-fft/numsim_fft.h>

#include <cmath>
#include <complex>
#include <cstdio>
#include <numbers>

int main() {
  using namespace numsim_fft;
  using vec = tmech::tensor<double, 3, 1>;
  using tensor2 = tmech::tensor<double, 3, 2>;
  using cvec = tmech::tensor<std::complex<double>, 3, 1>;
  using ctensor2 = tmech::tensor<std::complex<double>, 3, 2>;

  constexpr double pi{std::numbers::pi};
  extents<3> const grid{16, 16, 16};
  std::array<double, 3> const length{1.0, 2.0, 1.5};
  std::array<double, 3> h{};
  for (std::size_t d{0}; d < 3; ++d)
    h[d] = length[d] / static_cast<double>(grid[d]);

  // u = (sin(2 pi y / Ly), cos(2 pi z / Lz), sin(4 pi x / Lx))
  field<vec, 3> u{grid};
  field<tensor2, 3> eps_exact{grid};
  for (std::size_t i{0}; i < grid[0]; ++i)
    for (std::size_t j{0}; j < grid[1]; ++j)
      for (std::size_t k{0}; k < grid[2]; ++k) {
        double const x{i * h[0]}, y{j * h[1]}, z{k * h[2]};
        double const a{2 * pi / length[1]}, b{2 * pi / length[2]}, c{4 * pi / length[0]};
        u[i, j, k](0) = std::sin(a * y);
        u[i, j, k](1) = std::cos(b * z);
        u[i, j, k](2) = std::sin(c * x);
        tensor2 grad; // grad(p, q) = d u_p / d x_q
        grad(0, 1) = a * std::cos(a * y);
        grad(1, 2) = -b * std::sin(b * z);
        grad(2, 0) = c * std::cos(c * x);
        eps_exact[i, j, k] = tmech::sym(grad);
      }

  auto const plan{make_r2c_plan<double>(grid).value()};
  field<cvec, 3> u_hat{plan.spectral_extents()};
  if (!plan.forward(u, u_hat))
    return 1;

  field<ctensor2, 3> eps_hat{plan.spectral_extents()};
  auto const &se{plan.spectral_extents()};
  for (std::size_t p{0}; p < eps_hat.size(); ++p) {
    auto const idx{se.multi_index(p)};
    auto kv{wave_vector(plan, idx, h)};
    for (std::size_t d{0}; d < 3; ++d) // first derivative: drop Nyquist modes
      if (is_nyquist(idx[d], grid[d]))
        kv[d] = 0;
    auto const uh{u_hat.point(p)};
    auto e{eps_hat.point(p)};
    for (std::size_t a{0}; a < 3; ++a)
      for (std::size_t b{0}; b < 3; ++b)
        e(a, b) = std::complex<double>{0, 0.5} * (kv[b] * uh(a) + kv[a] * uh(b));
  }

  field<tensor2, 3> eps{grid};
  if (!plan.backward(eps_hat, eps))
    return 1;

  double max_err{0};
  for (std::size_t s{0}; s < eps.scalar_size(); ++s)
    max_err = std::max(max_err, std::abs(eps.data()[s] - eps_exact.data()[s]));
  std::printf("spectral strain: max |eps - eps_exact| = %.3e\n", max_err);
  return max_err < 1e-10 ? 0 : 1;
}
