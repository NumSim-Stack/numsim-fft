find_package(benchmark REQUIRED)

add_executable(numsim_fft_benchmark kernel_benchmark.cpp plan_benchmark.cpp)
target_link_libraries(numsim_fft_benchmark PRIVATE numsim-fft::numsim-fft benchmark::benchmark_main)
target_compile_options(numsim_fft_benchmark PRIVATE -O3 -march=native)

if(NUMSIM_FFT_ENABLE_MPI)
    # plain MPI program (mpiexec -n P ...), no Google Benchmark
    add_executable(numsim_fft_distributed_benchmark distributed_benchmark.cpp)
    target_link_libraries(numsim_fft_distributed_benchmark PRIVATE numsim-fft::numsim-fft)
    target_compile_options(numsim_fft_distributed_benchmark PRIVATE -O3 -march=native)
endif()
