add_numsim_fft_test(slab_decomposition_test SOURCES slab_decomposition_test.cpp transpose_sizes_test.cpp)

if(NUMSIM_FFT_ENABLE_MPI)
    add_numsim_fft_test(distributed_plan_test SOURCES distributed_plan_test.cpp
                        NO_MAIN MPI_PROCS 1 2 3 4)
endif()
