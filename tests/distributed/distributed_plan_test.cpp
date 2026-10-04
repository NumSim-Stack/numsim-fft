// Run with mpiexec on 1..4 ranks. Every rank builds the same global input,
// transforms its slab with the distributed plan and compares with the
// matching slab of the serial transform.

#include <numsim-fft/distributed/distributed_plan.h>
#include <numsim-fft/execution/openmp.h>

#include "common/random_fields.h"
#include "common/tolerances.h"

#include <gtest/gtest.h>
#include <mpl/mpl.hpp>

#include <complex>

using namespace numsim::fft;
using ak = axis_kind;

namespace {

mpl::communicator const &world() { return mpl::environment::comm_world(); }

template <typename E, std::size_t D> field<E, D> random_global(extents<D> const &e) {
  field<E, D> f{e};
  auto const v{test::random_vector<typename field<E, D>::scalar_type>(f.scalar_size(), 21)};
  std::ranges::copy(v, f.data());
  return f;
}

/// Sub-block [offset, offset + local) along `axis` of a global field.
template <typename E, std::size_t D, typename A>
field<E, D, A> slab(field<E, D, A> const &global, std::size_t axis, std::size_t offset,
                    std::size_t local) {
  auto const &ge{global.extents()};
  field<E, D, A> out{ge.with_extent(axis, local)};
  std::size_t const outer{ge.outer(axis)}, n{ge[axis]};
  std::size_t const inner{ge.inner(axis) * global.components};
  for (std::size_t o{0}; o < outer; ++o)
    for (std::size_t j{0}; j < local; ++j)
      std::copy_n(global.data() + (o * n + offset + j) * inner, inner,
                  out.data() + (o * local + j) * inner);
  return out;
}

/// Compares the distributed forward/backward with the serial plan.
template <typename EIn, typename EOut, std::size_t D, typename Exec = sequential_executor>
void check(extents<D> const &e, transform_domain domain, std::array<ak, D> const &kinds,
           plan_options const &options = {}, Exec const &exec = {}) {
  using T = real_type_t<typename field<EIn, D>::scalar_type>;
  auto const serial{plan<T, D>::create(e, domain, kinds, options).value()};
  auto const dist{distributed_plan<T, D>::create(world(), e, domain, kinds, options).value()};
  ASSERT_EQ(dist.global_spectral_extents(), serial.spectral_extents());

  auto const global_in{random_global<EIn>(e)};
  field<EOut, D> global_out{serial.spectral_extents()};
  ASSERT_TRUE(serial.forward(global_in, global_out).has_value());

  auto const local_in{slab(global_in, 0, dist.physical_offset(),
                           dist.local_physical_extents()[0])};
  field<EOut, D> local_out{dist.local_spectral_extents()};
  workspace<T> ws;
  ASSERT_TRUE(dist.forward(local_in, local_out, exec, ws).has_value());
  {
    // the workspace path and the temporary path agree exactly
    field<EOut, D> again{dist.local_spectral_extents()};
    ASSERT_TRUE(dist.forward(local_in, again, exec).has_value());
    EXPECT_TRUE(std::ranges::equal(again.scalars(), local_out.scalars()));
    ASSERT_TRUE(dist.forward(local_in, again, exec, ws).has_value()); // reuse
    EXPECT_TRUE(std::ranges::equal(again.scalars(), local_out.scalars()));
  }
  auto const expected{slab(global_out, 1, dist.spectral_offset(),
                           dist.local_spectral_extents()[1])};
  EXPECT_LT(test::relative_error(local_out.scalars(), expected.scalars()),
            test::tolerance<T>(e.size(), 16))
      << "rank " << world().rank();

  field<EIn, D> local_back{dist.local_physical_extents()};
  ASSERT_TRUE(dist.backward(local_out, local_back, exec, ws).has_value());
  EXPECT_LT(test::relative_error(local_back.scalars(), local_in.scalars()),
            test::tolerance<T>(e.size(), 16))
      << "rank " << world().rank();
}

using C = std::complex<double>;
using tensor2 = tmech::tensor<double, 3, 2>;
using ctensor2 = tmech::tensor<C, 3, 2>;

} // namespace

TEST(distributed_plan, load_and_store_hooks_act_on_the_local_points) {
  extents<3> const e{7, 6, 8};
  auto const dist{
      distributed_plan<double, 3>::create(world(), e, transform_domain::real_to_complex, {}).value()};
  auto const global_in{random_global<tensor2>(e)};
  auto const local_in{slab(global_in, 0, dist.physical_offset(), dist.local_physical_extents()[0])};
  auto const map = [](std::size_t point, double const *src, double *dst) {
    for (std::size_t c{0}; c < 9; ++c)
      dst[c] = static_cast<double>(point % 3 + 1) * src[c];
  };
  field<tensor2, 3> mapped{local_in.extents()};
  for (std::size_t p{0}; p < local_in.size(); ++p)
    map(p, local_in.data() + 9 * p, mapped.data() + 9 * p);
  workspace<double> ws;
  field<ctensor2, 3> plain{dist.local_spectral_extents()}, hooked{dist.local_spectral_extents()};
  ASSERT_TRUE(dist.forward(mapped, plain, sequential_executor{}, ws).has_value());
  ASSERT_TRUE(dist.forward(local_in, hooked, sequential_executor{}, ws, load_hook{map}).has_value());
  EXPECT_TRUE(std::ranges::equal(plain.scalars(), hooked.scalars()));

  struct counter {
    std::vector<int> visits;
    void begin(std::size_t) {}
    void operator()(std::size_t, std::size_t point, double const *) { ++visits[point]; }
  } count{std::vector<int>(local_in.size(), 0)};
  field<tensor2, 3> back{local_in.extents()}, back_hooked{local_in.extents()};
  ASSERT_TRUE(dist.backward(plain, back, sequential_executor{}, ws).has_value());
  ASSERT_TRUE(dist.backward(plain, back_hooked, sequential_executor{}, ws, store_hook{count}).has_value());
  EXPECT_TRUE(std::ranges::equal(back.scalars(), back_hooked.scalars()));
  for (int v : count.visits)
    EXPECT_EQ(v, 1);
}

TEST(distributed_plan, c2c_3d_periodic) {
  check<C, C>(extents{7, 6, 5}, transform_domain::complex_to_complex, {});
}

TEST(distributed_plan, c2c_2d_mixed_kinds) {
  check<C, C>(extents{5, 9}, transform_domain::complex_to_complex, {ak::dct2, ak::periodic});
}

TEST(distributed_plan, r2c_3d_rank2_tensor_field) {
  check<tensor2, ctensor2>(extents{6, 5, 8}, transform_domain::real_to_complex, {});
}

TEST(distributed_plan, r2c_with_half_spectrum_on_the_distributed_axis_0) {
  check<double, C>(extents{8, 5, 4}, transform_domain::real_to_complex,
                   {ak::periodic, ak::dct2, ak::dst1});
}

TEST(distributed_plan, r2c_with_half_spectrum_on_axis_1) {
  check<double, C>(extents{5, 7, 3}, transform_domain::real_to_complex,
                   {ak::dct3, ak::periodic, ak::dct1});
}

TEST(distributed_plan, r2r_3d) {
  check<double, double>(extents{5, 6, 3}, transform_domain::real_to_real,
                        {ak::dct1, ak::dst2, ak::dct4});
}

TEST(distributed_plan, more_ranks_than_rows) {
  check<C, C>(extents{2, 3, 4}, transform_domain::complex_to_complex, {});
  check<double, C>(extents{2, 3, 4}, transform_domain::real_to_complex, {});
}

TEST(distributed_plan, rank4_tensor_field_2d) {
  using tensor4 = tmech::tensor<double, 2, 4>;
  using ctensor4 = tmech::tensor<C, 2, 4>;
  check<tensor4, ctensor4>(extents{6, 7}, transform_domain::real_to_complex, {});
}

TEST(distributed_plan, ortho_normalization) {
  check<C, C>(extents{4, 6, 5}, transform_domain::complex_to_complex, {},
              {.norm = normalization::ortho});
}

TEST(distributed_plan, hybrid_mpi_openmp) {
  check<tensor2, ctensor2>(extents{8, 6, 6}, transform_domain::real_to_complex, {}, {},
                           openmp_executor{.num_threads = 2});
}

TEST(distributed_plan, local_extents_describe_the_slabs) {
  auto const p{make_distributed_r2c_plan<double>(world(), extents{10, 9, 8}).value()};
  slab_decomposition const rows{10, static_cast<std::size_t>(world().size())};
  slab_decomposition const cols{9, static_cast<std::size_t>(world().size())};
  auto const r{static_cast<std::size_t>(world().rank())};
  EXPECT_EQ(p.local_physical_extents(), (extents{rows.local_size(r), 9, 8}));
  EXPECT_EQ(p.local_spectral_extents(), (extents{10, cols.local_size(r), 5}));
  EXPECT_EQ(p.physical_offset(), rows.offset(r));
  EXPECT_EQ(p.spectral_offset(), cols.offset(r));
}

TEST(distributed_plan, extents_mismatch_is_reported) {
  auto const p{make_distributed_c2c_plan<double>(world(), extents{8, 8}).value()};
  field<C, 2> const wrong{extents{8, 8}};
  field<C, 2> out{p.local_spectral_extents()};
  if (world().size() > 1) {
    EXPECT_EQ(p.forward(wrong, out).error(), error::extents_mismatch);
  }
}

TEST(distributed_plan, error_on_one_rank_is_reported_on_all_ranks) {
  // Rank 1 passes a field with the wrong extents. Without agreement the
  // other ranks would block in the transpose; every rank must get the error.
  auto const p{make_distributed_c2c_plan<double>(world(), extents{8, 8}).value()};
  auto in_extents{p.local_physical_extents()};
  if (world().rank() == 1)
    in_extents = in_extents.with_extent(1, 7);
  field<C, 2> const in{in_extents};
  field<C, 2> out{p.local_spectral_extents()};
  if (world().size() > 1) {
    EXPECT_EQ(p.forward(in, out).error(), error::extents_mismatch);
    EXPECT_EQ(p.backward(out, const_cast<field<C, 2> &>(in)).error(), error::extents_mismatch);
  }
}

// --- Overlapped exchange (exchange_chunks > 1): same results as blocking ---

namespace {

/// The distributed forward/backward with k exchange chunks equals the
/// blocking one bit for bit (the local passes do the same arithmetic per
/// line) and the serial plan within the tolerance.
template <typename EIn, typename EOut, std::size_t D>
void check_overlap(extents<D> const &e, transform_domain domain, std::array<ak, D> const &kinds) {
  using T = real_type_t<typename field<EIn, D>::scalar_type>;
  for (std::size_t const k : {2u, 3u, 5u, 64u}) {
    SCOPED_TRACE(testing::Message() << "chunks " << k << ", rank " << world().rank());
    check<EIn, EOut>(e, domain, kinds, {.exchange_chunks = k});
    auto const blocking{distributed_plan<T, D>::create(world(), e, domain, kinds).value()};
    auto const overlapped{distributed_plan<T, D>::create(world(), e, domain, kinds, {.exchange_chunks = k}).value()};
    auto const global_in{random_global<EIn>(e)};
    auto const local_in{slab(global_in, 0, blocking.physical_offset(), blocking.local_physical_extents()[0])};
    field<EOut, D> a{blocking.local_spectral_extents()}, b{blocking.local_spectral_extents()};
    workspace<T> ws;
    ASSERT_TRUE(blocking.forward(local_in, a, sequential_executor{}, ws).has_value());
    ASSERT_TRUE(overlapped.forward(local_in, b, sequential_executor{}, ws).has_value());
    EXPECT_TRUE(std::ranges::equal(a.scalars(), b.scalars()));
    field<EIn, D> back_a{blocking.local_physical_extents()}, back_b{blocking.local_physical_extents()};
    ASSERT_TRUE(blocking.backward(a, back_a, sequential_executor{}, ws).has_value());
    ASSERT_TRUE(overlapped.backward(a, back_b, sequential_executor{}, ws).has_value());
    EXPECT_TRUE(std::ranges::equal(back_a.scalars(), back_b.scalars()));
  }
}

} // namespace

TEST(distributed_plan, overlapped_exchange_c2c) {
  check_overlap<C, C>(extents{9, 7, 6}, transform_domain::complex_to_complex, {});
}

TEST(distributed_plan, overlapped_exchange_r2c_tensor) {
  check_overlap<tensor2, ctensor2>(extents{8, 6, 6}, transform_domain::real_to_complex, {});
}

TEST(distributed_plan, overlapped_exchange_r2c_half_spectrum_on_axis_0) {
  check_overlap<double, C>(extents{9, 6, 5}, transform_domain::real_to_complex, {ak::periodic, ak::dct2, ak::dst1});
}

TEST(distributed_plan, overlapped_exchange_r2r_2d) {
  check_overlap<double, double>(extents{7, 5}, transform_domain::real_to_real, {ak::dct2, ak::dst1});
}

TEST(distributed_plan, overlapped_exchange_with_hooks_matches_blocking) {
  extents<3> const e{7, 6, 8};
  auto const blocking{distributed_plan<double, 3>::create(world(), e, transform_domain::real_to_complex, {}).value()};
  auto const overlapped{
      distributed_plan<double, 3>::create(world(), e, transform_domain::real_to_complex, {}, {.exchange_chunks = 3})
          .value()};
  auto const local_in{slab(random_global<tensor2>(e), 0, blocking.physical_offset(),
                           blocking.local_physical_extents()[0])};
  auto const map = [](std::size_t point, double const *src, double *dst) {
    for (std::size_t c{0}; c < 9; ++c)
      dst[c] = static_cast<double>(point % 3 + 1) * src[c];
  };
  workspace<double> ws;
  field<ctensor2, 3> a{blocking.local_spectral_extents()}, b{blocking.local_spectral_extents()};
  ASSERT_TRUE(blocking.forward(local_in, a, sequential_executor{}, ws, load_hook{map}).has_value());
  ASSERT_TRUE(overlapped.forward(local_in, b, sequential_executor{}, ws, load_hook{map}).has_value());
  EXPECT_TRUE(std::ranges::equal(a.scalars(), b.scalars()));
}

TEST(distributed_plan, is_move_only) {
  static_assert(!std::is_copy_constructible_v<distributed_plan<double, 3>>);
  static_assert(!std::is_copy_assignable_v<distributed_plan<double, 3>>);
  static_assert(std::is_nothrow_move_constructible_v<distributed_plan<double, 3>> ||
                std::is_move_constructible_v<distributed_plan<double, 3>>);
  auto p{make_distributed_c2c_plan<double>(world(), extents{4, 4, 4}).value()};
  auto q{std::move(p)};
  EXPECT_EQ(q.global_physical_extents(), (extents{4, 4, 4}));
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  if (world().rank() != 0) {
    auto &listeners{testing::UnitTest::GetInstance()->listeners()};
    delete listeners.Release(listeners.default_result_printer());
  }
  return RUN_ALL_TESTS();
}
