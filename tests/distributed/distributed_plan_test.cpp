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

using namespace numsim_fft;
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
  ASSERT_TRUE(dist.forward(local_in, local_out, exec).has_value());
  auto const expected{slab(global_out, 1, dist.spectral_offset(),
                           dist.local_spectral_extents()[1])};
  EXPECT_LT(test::relative_error(local_out.scalars(), expected.scalars()),
            test::tolerance<T>(e.size(), 16))
      << "rank " << world().rank();

  field<EIn, D> local_back{dist.local_physical_extents()};
  ASSERT_TRUE(dist.backward(local_out, local_back, exec).has_value());
  EXPECT_LT(test::relative_error(local_back.scalars(), local_in.scalars()),
            test::tolerance<T>(e.size(), 16))
      << "rank " << world().rank();
}

using C = std::complex<double>;
using tensor2 = tmech::tensor<double, 3, 2>;
using ctensor2 = tmech::tensor<C, 3, 2>;

} // namespace

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

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  if (world().rank() != 0) {
    auto &listeners{testing::UnitTest::GetInstance()->listeners()};
    delete listeners.Release(listeners.default_result_printer());
  }
  return RUN_ALL_TESTS();
}
