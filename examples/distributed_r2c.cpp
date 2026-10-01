// Distributed r2c transform of a rank-2 tensor field (slab decomposition).
// Run: mpiexec -n 4 ./distributed_r2c
// Checks Parseval's identity globally and the roundtrip locally.

#include <numsim-fft/numsim_fft.h>

#include <mpl/mpl.hpp>

#include <cmath>
#include <complex>
#include <cstdio>

int main() {
  using namespace numsim::fft;
  using tensor2 = tmech::tensor<double, 3, 2>;
  using ctensor2 = tmech::tensor<std::complex<double>, 3, 2>;

  auto const &comm{mpl::environment::comm_world()};
  extents<3> const grid{20, 18, 16};
  auto const plan{make_distributed_r2c_plan<double>(comm, grid).value()};

  // each rank fills its slab: rows physical_offset() .. + local extent 0
  field<tensor2, 3> eps{plan.local_physical_extents()};
  auto const &le{eps.extents()};
  for (std::size_t p{0}; p < eps.size(); ++p) {
    auto const idx{le.multi_index(p)};
    double const x{static_cast<double>(idx[0] + plan.physical_offset())};
    tensor2 t;
    t(0, 0) = std::sin(0.3 * x + 0.1 * static_cast<double>(idx[1]));
    t(1, 2) = t(2, 1) = std::cos(0.2 * static_cast<double>(idx[2]));
    eps.point(p) = t;
  }

  field<ctensor2, 3> eps_hat{plan.local_spectral_extents()};
  if (!plan.forward(eps, eps_hat))
    return 1;

  // Parseval with a half spectrum: modes 1 .. n/2-1 of the last axis count twice.
  double e_real{0}, e_spec{0};
  for (auto v : eps.scalars())
    e_real += v * v;
  auto const &se{eps_hat.extents()};
  std::size_t const n_last{grid[2]};
  for (std::size_t p{0}; p < eps_hat.size(); ++p) {
    std::size_t const k{se.multi_index(p)[2]};
    double const weight{(k == 0 || 2 * k == n_last) ? 1.0 : 2.0};
    for (std::size_t c{0}; c < eps_hat.components; ++c)
      e_spec += weight * std::norm(eps_hat.data()[p * eps_hat.components + c]);
  }
  comm.allreduce(mpl::plus<double>(), e_real);
  comm.allreduce(mpl::plus<double>(), e_spec);
  double const parseval{std::abs(e_spec / (static_cast<double>(grid.size()) * e_real) - 1.0)};

  field<tensor2, 3> back{plan.local_physical_extents()};
  if (!plan.backward(eps_hat, back))
    return 1;
  double err{0};
  for (std::size_t s{0}; s < back.scalar_size(); ++s)
    err = std::max(err, std::abs(back.data()[s] - eps.data()[s]));
  comm.allreduce(mpl::max<double>(), err);

  if (comm.rank() == 0)
    std::printf("distributed r2c on %d ranks: Parseval deviation %.3e, roundtrip error %.3e\n",
                comm.size(), parseval, err);
  return (parseval < 1e-12 && err < 1e-13) ? 0 : 1;
}
