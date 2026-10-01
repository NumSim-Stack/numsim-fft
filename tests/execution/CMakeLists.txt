add_numsim_fft_test(sequential_executor_test SOURCES sequential_test.cpp TSAN)

if(NUMSIM_FFT_ENABLE_OPENMP)
    add_numsim_fft_test(openmp_executor_test SOURCES openmp_test.cpp)
endif()

if(NUMSIM_FFT_ENABLE_HPX)
    add_numsim_fft_test(hpx_executor_test SOURCES hpx_test.cpp NO_MAIN NO_SANITIZERS)
endif()

# Several threads sharing one plan (sequential executor inside each thread),
# the case ThreadSanitizer can check without an instrumented OpenMP runtime.
add_numsim_fft_test(concurrency_test SOURCES concurrency_test.cpp TSAN)
