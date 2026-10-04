// Strong scaling of the distributed transforms: slab (distributed_plan)
// against pencils (pencil_plan, default grid and P0 x P1 given).
//
//   mpiexec -n P numsim_fft_distributed_benchmark [N] [repetitions] [P0 P1] [chunks]
//
// P0 = P1 = 0 picks the default grid. With chunks > 1 the overlapped
// exchange (plan_options::exchange_chunks) is timed next to the blocking one.
//
// Times one forward + backward r2c transform of a rank-2 tensor field on an
// N^3 grid (the pair a Lippmann-Schwinger iteration needs); prints the
// median over the repetitions of the slowest rank's time.

#include <numsim-fft/distributed/distributed_plan.h>
#include <numsim-fft/distributed/pencil_plan.h>

#include <tmech/tmech.h>

#include <mpl/mpl.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace numsim::fft;
using tensor2 = tmech::tensor<double, 3, 2>;
using ctensor2 = tmech::tensor<std::complex<double>, 3, 2>;

namespace {

template <typename Plan> double time_pair(mpl::communicator const &comm, Plan const &p, int repetitions) {
  field<tensor2, 3> x{p.local_physical_extents()};
  for (std::size_t i{0}; i < x.scalar_size(); ++i)
    x.data()[i] = static_cast<double>((i * 7919) % 1000) / 1000.0;
  field<ctensor2, 3> X{p.local_spectral_extents()};
  workspace<double> ws;
  (void)p.forward(x, X, sequential_executor{}, ws); // warm-up: buffers, layouts
  (void)p.backward(X, x, sequential_executor{}, ws);
  std::vector<double> times;
  for (int r{0}; r < repetitions; ++r) {
    comm.barrier();
    auto const t0{std::chrono::steady_clock::now()};
    if (!p.forward(x, X, sequential_executor{}, ws) || !p.backward(X, x, sequential_executor{}, ws))
      std::abort();
    double t{std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count()};
    comm.allreduce(mpl::max<double>(), t);
    times.push_back(t);
  }
  std::ranges::sort(times);
  return times[times.size() / 2];
}

} // namespace

int main(int argc, char **argv) {
  auto const &comm{mpl::environment::comm_world()};
  std::size_t const n{argc > 1 ? std::stoul(argv[1]) : 64};
  int const repetitions{argc > 2 ? std::stoi(argv[2]) : 9};
  std::array<std::size_t, 2> grid{0, 0};
  if (argc > 4)
    grid = {std::stoul(argv[3]), std::stoul(argv[4])};
  std::size_t const chunks{argc > 5 ? std::stoul(argv[5]) : 1};
  extents<3> const e{n, n, n};
  plan_options const overlapped{.exchange_chunks = chunks};

  auto const slab{make_distributed_r2c_plan<double>(comm, e).value()};
  double const t_slab{time_pair(comm, slab, repetitions)};
  auto const pencil{make_pencil_r2c_plan<double>(comm, e, {}, {}, grid).value()};
  double const t_pencil{time_pair(comm, pencil, repetitions)};
  double t_slab_k{0}, t_pencil_k{0};
  if (chunks > 1) {
    t_slab_k = time_pair(comm, make_distributed_r2c_plan<double>(comm, e, {}, overlapped).value(), repetitions);
    t_pencil_k = time_pair(comm, make_pencil_r2c_plan<double>(comm, e, {}, overlapped, grid).value(), repetitions);
  }
  if (comm.rank() == 0) {
    std::printf("N=%zu ranks=%d  slab %.4f s  pencil %zux%zu %.4f s", n, comm.size(), t_slab,
                pencil.process_grid()[0], pencil.process_grid()[1], t_pencil);
    if (chunks > 1)
      std::printf("  | %zu chunks: slab %.4f s  pencil %.4f s", chunks, t_slab_k, t_pencil_k);
    std::printf("\n");
  }
}
