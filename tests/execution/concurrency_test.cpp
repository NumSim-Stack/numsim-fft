// A plan is immutable after construction: several threads may run it at
// once, each with its own fields. Built with ThreadSanitizer as well.

#include <numsim-fft/numsim_fft.h>

#include "common/random_fields.h"
#include "common/tolerances.h"

#include <gtest/gtest.h>

#include <atomic>
#include <complex>
#include <thread>
#include <vector>

using namespace numsim::fft;

TEST(concurrency, threads_share_one_plan) {
  using tensor2 = tmech::tensor<double, 3, 2>;
  using ctensor2 = tmech::tensor<std::complex<double>, 3, 2>;
  extents<3> const e{12, 10, 14};
  std::array<axis_kind, 3> const kinds{axis_kind::dct2, axis_kind::periodic, axis_kind::periodic};
  auto const plan{make_r2c_plan<double>(e, kinds).value()};

  field<tensor2, 3> in{e};
  auto const v{test::random_vector<double>(in.scalar_size(), 5)};
  std::ranges::copy(v, in.data());

  // reference result from the main thread
  field<ctensor2, 3> ref_hat{plan.spectral_extents()};
  ASSERT_TRUE(plan.forward(in, ref_hat).has_value());

  std::atomic<int> failures{0};
  std::vector<std::thread> threads;
  for (int t{0}; t < 8; ++t)
    threads.emplace_back([&] {
      field<ctensor2, 3> hat{plan.spectral_extents()};
      field<tensor2, 3> back{e};
      for (int r{0}; r < 10; ++r) {
        if (!plan.forward(in, hat) || !plan.backward(hat, back))
          ++failures;
        if (!std::ranges::equal(hat.scalars(), ref_hat.scalars()))
          ++failures;
        if (test::relative_error(back.scalars(), in.scalars()) > 1e-13L)
          ++failures;
      }
    });
  for (auto &th : threads)
    th.join();
  EXPECT_EQ(failures.load(), 0);
}

TEST(concurrency, one_shot_api_from_threads) {
  using C = std::complex<double>;
  field<C, 2> x{extents{9, 7}};
  auto const v{test::random_vector<C>(x.scalar_size(), 8)};
  std::ranges::copy(v, x.data());
  auto const expected{fft(x).value()};
  std::atomic<int> failures{0};
  std::vector<std::thread> threads;
  for (int t{0}; t < 6; ++t)
    threads.emplace_back([&] {
      auto const X{fft(x)};
      if (!X || !std::ranges::equal(X->scalars(), expected.scalars()))
        ++failures;
    });
  for (auto &th : threads)
    th.join();
  EXPECT_EQ(failures.load(), 0);
}
