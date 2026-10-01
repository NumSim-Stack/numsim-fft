#ifndef NUMSIM_FFT_TESTS_EXECUTOR_TESTS_H
#define NUMSIM_FFT_TESTS_EXECUTOR_TESTS_H

// Typed tests shared by every executor backend: the including file defines
// the list `executor_types` before including this header.

#include <numsim-fft/numsim_fft.h>

#include "common/random_fields.h"

#include <gtest/gtest.h>

#include <atomic>
#include <complex>
#include <ranges>
#include <vector>

template <typename Exec> class executor_test : public testing::Test {};
TYPED_TEST_SUITE(executor_test, executor_types);

TYPED_TEST(executor_test, satisfies_the_concept) {
  static_assert(numsim_fft::executor<TypeParam>);
  EXPECT_GE(TypeParam{}.concurrency(), 1u);
}

TYPED_TEST(executor_test, bulk_runs_every_index_exactly_once) {
  TypeParam const exec{};
  std::size_t const n{1000};
  std::vector<std::atomic<int>> hits(n);
  exec.bulk(n, [&](std::size_t i) { hits[i].fetch_add(1); });
  for (std::size_t i{0}; i < n; ++i)
    ASSERT_EQ(hits[i].load(), 1) << i;
  exec.bulk(0, [&](std::size_t) { FAIL(); });
}

TYPED_TEST(executor_test, bulk_propagates_exceptions) {
  TypeParam const exec{};
  EXPECT_THROW(exec.bulk(64,
                         [](std::size_t i) {
                           if (i == 17)
                             throw std::runtime_error("boom");
                         }),
               std::runtime_error);
}

TYPED_TEST(executor_test, plan_results_are_bitwise_identical_to_sequential) {
  using namespace numsim_fft;
  using tensor2 = tmech::tensor<double, 3, 2>;
  using ctensor2 = tmech::tensor<std::complex<double>, 3, 2>;
  extents<3> const e{12, 10, 16};
  std::array<axis_kind, 3> const kinds{axis_kind::periodic, axis_kind::dct2, axis_kind::periodic};
  // small blocks so that every pass has many work items to distribute
  auto const p{make_r2c_plan<double>(e, kinds, {.max_batch = 4}).value()};

  field<tensor2, 3> in{e};
  auto const v{test::random_vector<double>(in.scalar_size(), 9)};
  std::ranges::copy(v, in.data());

  field<ctensor2, 3> seq_hat{p.spectral_extents()}, par_hat{p.spectral_extents()};
  ASSERT_TRUE(p.forward(in, seq_hat, sequential_executor{}).has_value());
  ASSERT_TRUE(p.forward(in, par_hat, TypeParam{}).has_value());
  EXPECT_TRUE(std::ranges::equal(seq_hat.scalars(), par_hat.scalars()));

  field<tensor2, 3> seq_back{e}, par_back{e};
  ASSERT_TRUE(p.backward(seq_hat, seq_back, sequential_executor{}).has_value());
  ASSERT_TRUE(p.backward(par_hat, par_back, TypeParam{}).has_value());
  EXPECT_TRUE(std::ranges::equal(seq_back.scalars(), par_back.scalars()));
}

#endif // NUMSIM_FFT_TESTS_EXECUTOR_TESTS_H
