// Plans borrow all temporaries from a caller-owned workspace. After a first
// (warming) call, further calls with the same workspace do not touch the
// heap: solvers call the transforms thousands of times.

#include <numsim-fft/numsim_fft.h>

#include "common/random_fields.h"
#include "common/tolerances.h"

#include <gtest/gtest.h>

#include <atomic>
#include <complex>
#include <cstdlib>
#include <new>

namespace {
std::atomic<std::size_t> g_allocations{0};
}

// Counting replacements for the global allocation functions (this TU only
// defines them; they serve the whole test binary).
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmismatched-new-delete" // false positive on replacements
#endif
void *operator new(std::size_t n) {
  ++g_allocations;
  if (void *p{std::malloc(n == 0 ? 1 : n)})
    return p;
  throw std::bad_alloc{};
}
void *operator new(std::size_t n, std::align_val_t al) {
  ++g_allocations;
  if (void *p{std::aligned_alloc(static_cast<std::size_t>(al),
                                 (n + static_cast<std::size_t>(al) - 1) /
                                     static_cast<std::size_t>(al) * static_cast<std::size_t>(al))})
    return p;
  throw std::bad_alloc{};
}
void operator delete(void *p) noexcept { std::free(p); }
void operator delete(void *p, std::size_t) noexcept { std::free(p); }
void operator delete(void *p, std::align_val_t) noexcept { std::free(p); }
void operator delete(void *p, std::size_t, std::align_val_t) noexcept { std::free(p); }
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

using namespace numsim_fft;
using ak = axis_kind;

namespace {
template <typename E, std::size_t D> field<E, D> random_field(extents<D> const &e) {
  field<E, D> f{e};
  auto const v{test::random_vector<typename field<E, D>::scalar_type>(f.scalar_size(), 17)};
  std::ranges::copy(v, f.data());
  return f;
}
} // namespace

TEST(workspace, results_are_identical_with_and_without_workspace) {
  using tensor2 = tmech::tensor<double, 3, 2>;
  using ctensor2 = tmech::tensor<std::complex<double>, 3, 2>;
  extents<3> const e{8, 6, 10};
  auto const p{make_r2c_plan<double>(e, {ak::dct2, ak::periodic, ak::periodic}).value()};
  auto const in{random_field<tensor2>(e)};
  field<ctensor2, 3> a{p.spectral_extents()}, b{p.spectral_extents()};
  field<tensor2, 3> back{e};
  workspace<double> ws;
  ASSERT_TRUE(p.forward(in, a).has_value());
  ASSERT_TRUE(p.forward(in, b, ws).has_value());
  EXPECT_TRUE(std::ranges::equal(a.scalars(), b.scalars()));
  ASSERT_TRUE(p.backward(b, back, sequential_executor{}, ws).has_value());
  EXPECT_LT(test::relative_error(back.scalars(), in.scalars()), 1e-14L);
}

TEST(workspace, warm_workspace_means_no_allocations) {
  using tensor2 = tmech::tensor<double, 3, 2>;
  using ctensor2 = tmech::tensor<std::complex<double>, 3, 2>;
  extents<3> const e{12, 10, 8};
  // r2c with other axes: the backward pass needs the input copy as well
  auto const p{make_r2c_plan<double>(e).value()};
  auto const in{random_field<tensor2>(e)};
  field<ctensor2, 3> hat{p.spectral_extents()};
  field<tensor2, 3> back{e};
  workspace<double> ws;
  std::size_t const cold{g_allocations.load()};
  ASSERT_TRUE(p.forward(in, hat, ws).has_value()); // warm-up allocates
  ASSERT_TRUE(p.backward(hat, back, ws).has_value());
  ASSERT_GT(g_allocations.load(), cold); // the counter does see allocations

  std::size_t const before{g_allocations.load()};
  for (int r{0}; r < 3; ++r) {
    ASSERT_TRUE(p.forward(in, hat, ws).has_value());
    ASSERT_TRUE(p.backward(hat, back, ws).has_value());
  }
  EXPECT_EQ(g_allocations.load() - before, 0u);
  EXPECT_LT(test::relative_error(back.scalars(), in.scalars()), 1e-14L);
}

TEST(workspace, serves_several_plans_and_grows) {
  using C = std::complex<double>;
  workspace<double> ws;
  auto const small{make_c2c_plan<double>(extents{4, 4}).value()};
  auto const big{make_c2c_plan<double>(extents{16, 12}, {ak::dst1, ak::periodic}).value()};
  field<C, 2> s_in{extents{4, 4}}, s_out{extents{4, 4}}, b_in{extents{16, 12}},
      b_out{extents{16, 12}}, b_ref{extents{16, 12}};
  ASSERT_TRUE(small.forward(s_in, s_out, ws).has_value());
  auto const v{test::random_vector<C>(b_in.scalar_size(), 2)};
  std::ranges::copy(v, b_in.data());
  ASSERT_TRUE(big.forward(b_in, b_out, ws).has_value());
  ASSERT_TRUE(big.forward(b_in, b_ref).has_value());
  EXPECT_TRUE(std::ranges::equal(b_out.scalars(), b_ref.scalars()));
  ASSERT_TRUE(small.forward(s_in, s_out, ws).has_value()); // back to the small one
}

TEST(workspace, one_shot_api_accepts_a_workspace) {
  using C = std::complex<double>;
  workspace<double> ws;
  field<C, 2> x{extents{6, 5}};
  auto const X{fft(x, {}, {}, sequential_executor{}, &ws)};
  ASSERT_TRUE(X.has_value());
}
