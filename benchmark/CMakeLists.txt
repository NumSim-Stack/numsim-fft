find_package(benchmark REQUIRED)

add_executable(numsim_fft_benchmark kernel_benchmark.cpp plan_benchmark.cpp)
target_link_libraries(numsim_fft_benchmark PRIVATE numsim-fft::numsim-fft benchmark::benchmark_main)
target_compile_options(numsim_fft_benchmark PRIVATE -O3 -march=native)
