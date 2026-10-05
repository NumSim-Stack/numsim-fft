// Pencil decomposition (#3). Run with mpiexec on 1..8 ranks: every rank builds
// the same global input, transforms its pencil with pencil_plan on several
// process grids and compares with the matching block of the serial transform.

#include <numsim-fft/distributed/distributed_plan.h>
#include <numsim-fft/distributed/pencil_plan.h>
#include <numsim-fft/execution/openmp.h>

#include "common/random_fields.h"
#include "common/tolerances.h"

#include <gtest/gtest.h>
#include <mpl/mpl.hpp>

#include <array>
#include <complex>
#include <vector>

using namespace numsim::fft;
using ak = axis_kind;
using grid2 = std::array<std::size_t, 2>;

namespace {

mpl::communicator const &world() { return mpl::environment::comm_world(); }
std::size_t ranks() { return static_cast<std::size_t>(world().size()); }

using C = std::complex<double>;
using tensor2 = tmech::tensor<double, 3, 2>;
using ctensor2 = tmech::tensor<C, 3, 2>;

template <typename E> field<E, 3> random_global(extents<3> const &e) {
  field<E, 3> f{e};
  auto const v{test::random_vector<typename field<E, 3>::scalar_type>(f.scalar_size(), 33)};
  std::ranges::copy(v, f.data());
  return f;
}

/// Block [offset, offset + local) on every axis of a global field.
template <typename E, typename A>
field<E, 3, A> block(field<E, 3, A> const &g, std::array<std::size_t, 3> const &offset,
                     extents<3> const &local) {
  field<E, 3, A> out{local};
  auto const &ge{g.extents()};
  std::size_t const R{g.components};
  for (std::size_t i{0}; i < local[0]; ++i)
    for (std::size_t j{0}; j < local[1]; ++j)
      std::copy_n(g.data() + (((offset[0] + i) * ge[1] + offset[1] + j) * ge[2] + offset[2]) * R,
                  local[2] * R, out.data() + ((i * local[1] + j) * local[2]) * R);
  return out;
}

/// Grids to try on the current number of ranks: the default and every
/// factorisation P0 x P1 = P.
std::vector<grid2> grids() {
  std::vector<grid2> g{{0, 0}};
  for (std::size_t p0{1}; p0 <= ranks(); ++p0)
    if (ranks() % p0 == 0)
      g.push_back({p0, ranks() / p0});
  return g;
}

template <typename EIn, typename EOut, typename Exec = sequential_executor>
void check(extents<3> const &e, transform_domain domain, std::array<ak, 3> const &kinds,
           plan_options const &options = {}, Exec const &exec = {}) {
  using T = real_type_t<typename field<EIn, 3>::scalar_type>;
  auto const serial{plan<T, 3>::create(e, domain, kinds, options).value()};
  auto const global_in{random_global<EIn>(e)};
  field<EOut, 3> global_out{serial.spectral_extents()};
  ASSERT_TRUE(serial.forward(global_in, global_out).has_value());

  for (auto const grid : grids()) {
    SCOPED_TRACE(testing::Message() << "grid " << grid[0] << " x " << grid[1] << ", rank " << world().rank());
    auto const p{pencil_plan<T>::create(world(), e, domain, kinds, options, grid).value()};
    ASSERT_EQ(p.global_spectral_extents(), serial.spectral_extents());
    ASSERT_EQ(p.process_grid()[0] * p.process_grid()[1], ranks());
    auto const local_in{block(global_in, p.physical_offsets(), p.local_physical_extents())};
    field<EOut, 3> local_out{p.local_spectral_extents()};
    workspace<T> ws;
    ASSERT_TRUE(p.forward(local_in, local_out, exec, ws).has_value());
    field<EOut, 3> again{p.local_spectral_extents()};
    ASSERT_TRUE(p.forward(local_in, again, exec).has_value()); // without workspace
    EXPECT_TRUE(std::ranges::equal(again.scalars(), local_out.scalars()));
    auto const expected{block(global_out, p.spectral_offsets(), p.local_spectral_extents())};
    EXPECT_LT(test::relative_error(local_out.scalars(), expected.scalars()), test::tolerance<T>(e.size(), 16));

    field<EIn, 3> back{p.local_physical_extents()};
    ASSERT_TRUE(p.backward(local_out, back, exec, ws).has_value());
    EXPECT_LT(test::relative_error(back.scalars(), local_in.scalars()), test::tolerance<T>(e.size(), 16));
  }
}

} // namespace

TEST(pencil_plan, c2c_periodic_uneven) {
  check<C, C>(extents{7, 6, 5}, transform_domain::complex_to_complex, {});
}

TEST(pencil_plan, r2c_rank2_tensor_half_spectrum_on_axis_2) {
  check<tensor2, ctensor2>(extents{8, 6, 6}, transform_domain::real_to_complex, {});
}

TEST(pencil_plan, r2c_half_spectrum_on_axis_1) {
  check<double, C>(extents{5, 8, 6}, transform_domain::real_to_complex, {ak::periodic, ak::periodic, ak::dct2});
}

TEST(pencil_plan, r2c_half_spectrum_on_axis_0) {
  check<double, C>(extents{9, 6, 5}, transform_domain::real_to_complex, {ak::periodic, ak::dct2, ak::dst1});
}

TEST(pencil_plan, r2r_mixed_kinds) {
  check<double, double>(extents{6, 5, 7}, transform_domain::real_to_real, {ak::dct1, ak::dst2, ak::dct4});
}

TEST(pencil_plan, c2c_with_dct_axis) {
  check<C, C>(extents{4, 7, 6}, transform_domain::complex_to_complex, {ak::periodic, ak::dct2, ak::periodic});
}

TEST(pencil_plan, more_ranks_than_points) {
  check<C, C>(extents{2, 2, 3}, transform_domain::complex_to_complex, {});
  check<double, C>(extents{2, 3, 2}, transform_domain::real_to_complex, {});
}

TEST(pencil_plan, ortho_normalization) {
  check<C, C>(extents{4, 6, 5}, transform_domain::complex_to_complex, {}, {.norm = normalization::ortho});
}

TEST(pencil_plan, hybrid_mpi_openmp) {
  check<tensor2, ctensor2>(extents{8, 6, 6}, transform_domain::real_to_complex, {}, {},
                           openmp_executor{.num_threads = 2});
}

TEST(pencil_plan, local_extents_and_offsets_describe_the_pencils) {
  auto const p{make_pencil_r2c_plan<double>(world(), extents{10, 9, 8}).value()};
  auto const [P0, P1]{p.process_grid()};
  ASSERT_EQ(P0 * P1, ranks());
  auto const r{static_cast<std::size_t>(world().rank())};
  std::size_t const r0{r / P1}, r1{r % P1};
  slab_decomposition const d0{10, P0}, d1{9, P1}, e1{9, P0}, e2{5, P1};
  EXPECT_EQ(p.local_physical_extents(), (extents{d0.local_size(r0), d1.local_size(r1), 8}));
  EXPECT_EQ(p.local_spectral_extents(), (extents{10, e1.local_size(r0), e2.local_size(r1)}));
  EXPECT_EQ(p.physical_offsets(), (std::array<std::size_t, 3>{d0.offset(r0), d1.offset(r1), 0}));
  EXPECT_EQ(p.spectral_offsets(), (std::array<std::size_t, 3>{0, e1.offset(r0), e2.offset(r1)}));
}

TEST(pencil_plan, the_default_grid_is_one_row_while_the_ranks_fit_axis_0) {
  // One exchange instead of two: measured faster on a node as long as
  // every rank gets a slab.
  auto const p{make_pencil_c2c_plan<double>(world(), extents{8, 8, 8}).value()};
  EXPECT_EQ(p.process_grid(), (grid2{ranks(), 1}));
}

TEST(pencil_plan, with_more_ranks_than_slabs_the_default_grid_splits_axis_1_too) {
  // N0 = 2: P0 is the largest divisor of P not above 2
  auto const p{make_pencil_c2c_plan<double>(world(), extents{2, 8, 8}).value()};
  auto const [P0, P1]{p.process_grid()};
  EXPECT_EQ(P0 * P1, ranks());
  EXPECT_EQ(P0, ranks() % 2 == 0 ? std::min<std::size_t>(2, ranks()) : 1u);
}

TEST(pencil_plan, a_grid_that_does_not_cover_the_ranks_is_rejected) {
  auto const p{pencil_plan<double>::create(world(), extents{4, 4, 4}, transform_domain::complex_to_complex, {}, {},
                                           grid2{ranks() + 1, 1})};
  ASSERT_FALSE(p.has_value());
  EXPECT_EQ(p.error(), error::invalid_process_grid);
}

TEST(pencil_plan, errors_are_agreed_on_by_all_ranks) {
  auto const p{make_pencil_c2c_plan<double>(world(), extents{6, 6, 6}).value()};
  auto in_extents{p.local_physical_extents()};
  if (static_cast<std::size_t>(world().rank()) == ranks() - 1)
    in_extents = in_extents.with_extent(2, 5);
  field<C, 3> const in{in_extents};
  field<C, 3> out{p.local_spectral_extents()};
  EXPECT_EQ(p.forward(in, out).error(), error::extents_mismatch);
}

TEST(pencil_plan, load_and_store_hooks_act_on_the_local_points) {
  auto const p{make_pencil_r2c_plan<double>(world(), extents{7, 6, 8}).value()};
  auto const local_in{
      block(random_global<tensor2>(extents{7, 6, 8}), p.physical_offsets(), p.local_physical_extents())};
  auto const map = [](std::size_t point, double const *src, double *dst) {
    for (std::size_t c{0}; c < 9; ++c)
      dst[c] = static_cast<double>(point % 3 + 1) * src[c];
  };
  field<tensor2, 3> mapped{local_in.extents()};
  for (std::size_t q{0}; q < local_in.size(); ++q)
    map(q, local_in.data() + 9 * q, mapped.data() + 9 * q);
  workspace<double> ws;
  field<ctensor2, 3> plain{p.local_spectral_extents()}, hooked{p.local_spectral_extents()};
  ASSERT_TRUE(p.forward(mapped, plain, sequential_executor{}, ws).has_value());
  ASSERT_TRUE(p.forward(local_in, hooked, sequential_executor{}, ws, load_hook{map}).has_value());
  EXPECT_TRUE(std::ranges::equal(plain.scalars(), hooked.scalars()));

  struct counter {
    std::vector<int> visits;
    void begin(std::size_t) {}
    void operator()(std::size_t, std::size_t point, double const *) { ++visits[point]; }
  } count{std::vector<int>(local_in.size(), 0)};
  field<tensor2, 3> back{local_in.extents()}, back_hooked{local_in.extents()};
  ASSERT_TRUE(p.backward(plain, back, sequential_executor{}, ws).has_value());
  ASSERT_TRUE(p.backward(plain, back_hooked, sequential_executor{}, ws, store_hook{count}).has_value());
  EXPECT_TRUE(std::ranges::equal(back.scalars(), back_hooked.scalars()));
  for (int v : count.visits)
    EXPECT_EQ(v, 1);
}

// --- Overlapped exchange (exchange_chunks > 1): same results as blocking ---

namespace {

template <typename EIn, typename EOut>
void check_overlap(extents<3> const &e, transform_domain domain, std::array<ak, 3> const &kinds) {
  using T = real_type_t<typename field<EIn, 3>::scalar_type>;
  auto const global_in{random_global<EIn>(e)};
  for (auto const grid : grids())
    for (std::size_t const k : {2u, 3u, 7u}) {
      SCOPED_TRACE(testing::Message() << "grid " << grid[0] << " x " << grid[1] << ", chunks " << k << ", rank "
                                      << world().rank());
      auto const blocking{pencil_plan<T>::create(world(), e, domain, kinds, {}, grid).value()};
      auto const overlapped{pencil_plan<T>::create(world(), e, domain, kinds, {.exchange_chunks = k}, grid).value()};
      auto const local_in{block(global_in, blocking.physical_offsets(), blocking.local_physical_extents())};
      field<EOut, 3> a{blocking.local_spectral_extents()}, b{blocking.local_spectral_extents()};
      workspace<T> ws;
      ASSERT_TRUE(blocking.forward(local_in, a, sequential_executor{}, ws).has_value());
      ASSERT_TRUE(overlapped.forward(local_in, b, sequential_executor{}, ws).has_value());
      EXPECT_TRUE(std::ranges::equal(a.scalars(), b.scalars()));
      field<EIn, 3> back_a{blocking.local_physical_extents()}, back_b{blocking.local_physical_extents()};
      ASSERT_TRUE(blocking.backward(a, back_a, sequential_executor{}, ws).has_value());
      ASSERT_TRUE(overlapped.backward(a, back_b, sequential_executor{}, ws).has_value());
      EXPECT_TRUE(std::ranges::equal(back_a.scalars(), back_b.scalars()));
    }
}

} // namespace

TEST(pencil_plan, overlapped_exchange_c2c) {
  check_overlap<C, C>(extents{7, 6, 5}, transform_domain::complex_to_complex, {});
}

TEST(pencil_plan, overlapped_exchange_r2c_tensor_axis_2) {
  check_overlap<tensor2, ctensor2>(extents{8, 6, 6}, transform_domain::real_to_complex, {});
}

TEST(pencil_plan, overlapped_exchange_r2c_axis_1) {
  check_overlap<double, C>(extents{5, 8, 6}, transform_domain::real_to_complex, {ak::periodic, ak::periodic, ak::dct2});
}

TEST(pencil_plan, overlapped_exchange_r2c_axis_0) {
  check_overlap<double, C>(extents{9, 6, 5}, transform_domain::real_to_complex, {ak::periodic, ak::dct2, ak::dst1});
}

TEST(pencil_plan, overlapped_exchange_r2r) {
  check_overlap<double, double>(extents{6, 5, 7}, transform_domain::real_to_real, {ak::dct1, ak::dst2, ak::dct4});
}

TEST(pencil_plan, overlapped_exchange_against_the_serial_plan) {
  for (std::size_t const k : {2u, 4u})
    check<tensor2, ctensor2>(extents{8, 7, 6}, transform_domain::real_to_complex, {}, {.exchange_chunks = k});
}

TEST(pencil_plan, is_move_only) {
  static_assert(!std::is_copy_constructible_v<pencil_plan<double>>);
  auto p{make_pencil_c2c_plan<double>(world(), extents{4, 4, 4}).value()};
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

namespace {
/// Forward with `ws` on `comm` and grid, compared with the serial transform.
void forward_matches_serial(mpl::communicator const &comm, extents<3> const &e, grid2 grid, workspace<double> &ws) {
  auto const serial{plan<double, 3>::create(e, transform_domain::complex_to_complex, {}).value()};
  auto const global_in{random_global<C>(e)};
  field<C, 3> global_out{serial.spectral_extents()};
  ASSERT_TRUE(serial.forward(global_in, global_out).has_value());
  auto const p{pencil_plan<double>::create(comm, e, transform_domain::complex_to_complex, {}, {}, grid).value()};
  auto const local_in{block(global_in, p.physical_offsets(), p.local_physical_extents())};
  field<C, 3> local_out{p.local_spectral_extents()};
  ASSERT_TRUE(p.forward(local_in, local_out, sequential_executor{}, ws).has_value());
  auto const expected{block(global_out, p.spectral_offsets(), p.local_spectral_extents())};
  EXPECT_LT(test::relative_error(local_out.scalars(), expected.scalars()), test::tolerance<double>(e.size(), 16))
      << "grid " << grid[0] << " x " << grid[1] << ", rank " << world().rank();
}
} // namespace

TEST(pencil_plan, plans_on_differently_ordered_communicators_share_a_workspace) {
  int const P{world().size()};
  mpl::communicator const reversed{mpl::communicator::split, world(), 0, P - 1 - world().rank()};
  for (auto const grid : grids()) {
    workspace<double> ws;
    forward_matches_serial(world(), {8, 6, 5}, grid, ws);
    forward_matches_serial(reversed, {8, 6, 5}, grid, ws);
    forward_matches_serial(world(), {8, 6, 5}, grid, ws);
  }
}

TEST(pencil_plan, plans_of_different_grids_share_a_workspace) {
  for (auto const grid : grids()) {
    workspace<double> ws;
    forward_matches_serial(world(), {8, 6, 5}, grid, ws);
    forward_matches_serial(world(), {11, 7, 5}, grid, ws);
  }
}

TEST(pencil_plan, exchange_chunks_that_differ_between_ranks_are_rejected_on_every_rank) {
  plan_options options;
  options.exchange_chunks = static_cast<std::size_t>(world().rank()) + 1; // differs when P > 1
  auto const p{pencil_plan<double>::create(world(), {8, 6, 5}, transform_domain::complex_to_complex, {}, options)};
  auto const d{distributed_plan<double, 3>::create(world(), {8, 6, 5}, transform_domain::complex_to_complex, {}, options)};
  if (world().size() == 1) {
    EXPECT_TRUE(p.has_value());
    EXPECT_TRUE(d.has_value());
  } else {
    ASSERT_FALSE(p.has_value());
    EXPECT_EQ(p.error(), error::options_mismatch);
    ASSERT_FALSE(d.has_value());
    EXPECT_EQ(d.error(), error::options_mismatch);
  }
}
