// A constant rank-4 stiffness commutes with the Fourier transform:
//   sigma = C : eps  <=>  sigma_hat = C : eps_hat.
// Applies C pointwise in Fourier space with tmech::dcontract on complex
// tensors and compares with the real-space product.

#include <numsim-fft/numsim_fft.h>

#include <cmath>
#include <complex>
#include <cstdio>
#include <random>

int main() {
  using namespace numsim_fft;
  using tensor2 = tmech::tensor<double, 3, 2>;
  using tensor4 = tmech::tensor<double, 3, 4>;
  using ctensor2 = tmech::tensor<std::complex<double>, 3, 2>;
  using ctensor4 = tmech::tensor<std::complex<double>, 3, 4>;

  // isotropic stiffness, Lame parameters lambda and mu
  tensor2 const I{tmech::eye<double, 3, 2>()};
  double const lambda{120.0}, mu{80.0};
  tensor4 const C{lambda * tmech::otimes(I, I) + mu * (tmech::otimesu(I, I) + tmech::otimesl(I, I))};
  ctensor4 Cc;
  for (std::size_t i{0}; i < tensor4::size(); ++i)
    Cc.raw_data()[i] = C.raw_data()[i];

  // random symmetric strain field on a non-cubic grid
  extents<3> const grid{12, 10, 9};
  field<tensor2, 3> eps{grid};
  std::mt19937_64 gen{7};
  std::uniform_real_distribution<double> dist{-1e-3, 1e-3};
  for (std::size_t p{0}; p < eps.size(); ++p) {
    tensor2 a;
    for (std::size_t i{0}; i < 9; ++i)
      a.raw_data()[i] = dist(gen);
    eps.point(p) = tmech::sym(a);
  }

  field<tensor2, 3> sigma{grid};
  for (std::size_t p{0}; p < eps.size(); ++p)
    sigma.point(p) = tmech::dcontract(C, eps.point(p));

  auto const plan{make_r2c_plan<double>(grid).value()};
  field<ctensor2, 3> eps_hat{plan.spectral_extents()};
  if (!plan.forward(eps, eps_hat))
    return 1;
  for (std::size_t p{0}; p < eps_hat.size(); ++p)
    eps_hat.point(p) = tmech::eval(tmech::dcontract(Cc, eps_hat.point(p)));

  field<tensor2, 3> sigma_fft{grid};
  if (!plan.backward(eps_hat, sigma_fft))
    return 1;

  double num{0}, den{0};
  for (std::size_t s{0}; s < sigma.scalar_size(); ++s) {
    double const d{sigma_fft.data()[s] - sigma.data()[s]};
    num += d * d;
    den += sigma.data()[s] * sigma.data()[s];
  }
  double const rel{std::sqrt(num / den)};
  std::printf("fourier operator: ||F^-1(C : F(eps)) - C : eps|| / ||C : eps|| = %.3e\n", rel);
  return rel < 1e-13 ? 0 : 1;
}
