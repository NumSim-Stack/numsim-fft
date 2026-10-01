// 1D kernel throughput: GFlop/s with the 5 N log2 N convention.
#include <numsim-fft/kernel/c2c_plan_1d.h>
#include <numsim-fft/kernel/r2c_plan_1d.h>

#include <benchmark/benchmark.h>

#include <cmath>
#include <vector>

using namespace numsim_fft::kernel;

static void c2c_1d(benchmark::State &state) {
  std::size_t const n{static_cast<std::size_t>(state.range(0))};
  std::size_t const B{static_cast<std::size_t>(state.range(1))};
  c2c_plan_1d<double> const plan{n};
  std::vector<cplx<double>> data(n * B, {1, 0}), scratch(plan.scratch_size(B));
  for (auto _ : state) {
    plan.forward(data.data(), B, data.data(), B, B, scratch.data());
    benchmark::DoNotOptimize(data.data());
  }
  double const flops{5.0 * static_cast<double>(n) * std::log2(static_cast<double>(n)) *
                     static_cast<double>(B)};
  state.counters["GFlop/s"] = benchmark::Counter(flops * 1e-9, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(c2c_1d)->Args({128, 1})->Args({128, 9})->Args({128, 64})->Args({1024, 9})->Args({1024, 64})
    ->Args({1000, 64})->Args({101, 64});

static void r2c_1d(benchmark::State &state) {
  std::size_t const n{static_cast<std::size_t>(state.range(0))};
  std::size_t const B{static_cast<std::size_t>(state.range(1))};
  r2c_plan_1d<double> const plan{n};
  std::vector<double> in(n * B, 1.0);
  std::vector<cplx<double>> out(plan.spectrum_size() * B), scratch(plan.scratch_size(B));
  for (auto _ : state) {
    plan.forward(in.data(), B, out.data(), B, B, scratch.data());
    benchmark::DoNotOptimize(out.data());
  }
  double const flops{2.5 * static_cast<double>(n) * std::log2(static_cast<double>(n)) *
                     static_cast<double>(B)};
  state.counters["GFlop/s"] = benchmark::Counter(flops * 1e-9, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(r2c_1d)->Args({128, 9})->Args({1024, 64});
