// Whole-field transforms of a rank-2 tensor field (9 components).
#include <numsim-fft/numsim_fft.h>

#include <benchmark/benchmark.h>

using namespace numsim::fft;

template <typename Exec> static void r2c_rank2_3d(benchmark::State &state) {
  std::size_t const n{static_cast<std::size_t>(state.range(0))};
  field<tmech::tensor<double, 3, 2>, 3> eps{extents{n, n, n}, tmech::eye<double, 3, 2>()};
  auto const plan{make_r2c_plan<double>(eps.extents()).value()};
  field<tmech::tensor<std::complex<double>, 3, 2>, 3> hat{plan.spectral_extents()};
  Exec const exec{};
  for (auto _ : state) {
    (void)plan.forward(eps, hat, exec);
    (void)plan.backward(hat, eps, exec);
    benchmark::DoNotOptimize(eps.data());
  }
  state.counters["Mpoints/s"] = benchmark::Counter(
      static_cast<double>(eps.size()) * 1e-6, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(r2c_rank2_3d<sequential_executor>)->Arg(32)->Arg(64)->Arg(128)->Unit(benchmark::kMillisecond);
#if defined(NUMSIM_FFT_HAS_OPENMP)
BENCHMARK(r2c_rank2_3d<openmp_executor>)->Arg(64)->Arg(128)->Unit(benchmark::kMillisecond);
#endif
