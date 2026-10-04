// First touch (#2): a field constructed with an executor is zeroed by the
// executor's workers, in the chunks the field algebra uses, so on NUMA
// machines each page lands on the node of the thread that later uses it.

#include <numsim-fft/execution/sequential.h>
#include <numsim-fft/field/algebra.h>
#include <numsim-fft/field/field.h>

#include <tmech/tmech.h>

#include <gtest/gtest.h>

#include <complex>
#include <cstddef>
#include <cstring>
#include <mutex>
#include <thread>
#include <type_traits>
#include <vector>

using namespace numsim::fft;

namespace {

/// Runs every item on its own std::thread (real concurrency, no OpenMP).
struct thread_executor {
  std::size_t workers{4};
  std::size_t concurrency() const noexcept { return workers; }
  template <typename F> void bulk(std::size_t n, F &&f) const {
    std::vector<std::thread> threads;
    for (std::size_t i{0}; i < n; ++i)
      threads.emplace_back([&f, i] { f(i); });
    for (auto &t : threads)
      t.join();
  }
};

/// Records the bulk calls and runs the items in order.
struct recording_executor {
  std::size_t workers{4};
  mutable std::vector<std::size_t> calls;
  std::size_t concurrency() const noexcept { return workers; }
  template <typename F> void bulk(std::size_t n, F &&f) const {
    calls.push_back(n);
    for (std::size_t i{0}; i < n; ++i)
      f(i);
  }
};

/// Claims workers but never runs an item.
struct idle_executor {
  std::size_t concurrency() const noexcept { return 4; }
  template <typename F> void bulk(std::size_t, F &&) const {}
};

/// Aligned allocator that fills new memory with 0xAB, to see which bytes a
/// constructor writes.
template <typename T> struct poisoning_allocator : aligned_allocator<T> {
  using value_type = T;
  template <typename U> struct rebind {
    using other = poisoning_allocator<U>;
  };
  poisoning_allocator() = default;
  template <typename U> poisoning_allocator(poisoning_allocator<U> const &) noexcept {}
  T *allocate(std::size_t n) {
    T *p{aligned_allocator<T>::allocate(n)};
    std::memset(static_cast<void *>(p), 0xAB, n * sizeof(T));
    return p;
  }
  template <typename U> bool operator==(poisoning_allocator<U> const &) const noexcept { return true; }
};

template <typename F> bool all_scalars_zero(F const &f) {
  for (auto const s : f.scalars())
    if (s != typename F::scalar_type{})
      return false;
  return true;
}

} // namespace

TEST(first_touch, a_field_built_with_an_executor_is_zero) {
  field<tmech::tensor<double, 3, 2>, 3> real{extents{9, 7, 5}, thread_executor{}};
  EXPECT_TRUE(all_scalars_zero(real));
  field<std::complex<float>, 2> spectrum{extents{33, 17}, thread_executor{3}};
  EXPECT_TRUE(all_scalars_zero(spectrum));
  field<double, 1> sequential{extents<1>{1000}, sequential_executor{}};
  EXPECT_TRUE(all_scalars_zero(sequential));
}

TEST(first_touch, the_zeroing_runs_in_the_chunks_of_the_field_algebra) {
  recording_executor exec{.workers = 4, .calls = {}};
  field<tmech::tensor<double, 3, 2>, 3> f{extents{16, 16, 16}, exec};
  ASSERT_EQ(exec.calls.size(), 1u);
  std::size_t const touch_chunks{exec.calls[0]};
  fill(f, 1.0, exec); // the algebra's chunking, for comparison
  ASSERT_EQ(exec.calls.size(), 2u);
  EXPECT_EQ(touch_chunks, exec.calls[1]);
}

TEST(first_touch, only_the_executor_writes_the_new_memory) {
  using poisoned = field<double, 1, poisoning_allocator<double>>;
  poisoned untouched{extents<1>{64}, idle_executor{}};
  for (auto const s : untouched.scalars()) {
    unsigned char bytes[sizeof(double)];
    std::memcpy(bytes, &s, sizeof s);
    EXPECT_EQ(bytes[0], 0xAB);
  }
  poisoned zeroed{extents<1>{64}};
  EXPECT_TRUE(all_scalars_zero(zeroed)); // the plain constructor still zeroes
}

TEST(first_touch, the_allocator_type_and_copies_are_unchanged) {
  using F = field<tmech::tensor<double, 3, 2>, 2>;
  static_assert(std::is_same_v<F::allocator_type, aligned_allocator<double>>);
  F a{extents{4, 4}, thread_executor{}};
  a.scalars()[5] = 2.5;
  F const b{a};
  EXPECT_EQ(b.scalars()[5], 2.5);
  EXPECT_TRUE((std::is_same_v<decltype(b.get_allocator()), aligned_allocator<double>>));
}
