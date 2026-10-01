function(add_numsim_fft_example name)
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} PRIVATE numsim-fft::numsim-fft)
    target_compile_options(${name} PRIVATE $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>)
    if(NUMSIM_FFT_BUILD_TESTS)
        add_test(NAME example_${name} COMMAND ${name})
    endif()
endfunction()

add_numsim_fft_example(spectral_strain)
add_numsim_fft_example(fourier_operator)

if(NUMSIM_FFT_ENABLE_MPI)
    add_executable(distributed_r2c distributed_r2c.cpp)
    target_link_libraries(distributed_r2c PRIVATE numsim-fft::numsim-fft)
    if(NUMSIM_FFT_BUILD_TESTS)
        add_test(NAME example_distributed_r2c
                 COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 3 --oversubscribe
                         $<TARGET_FILE:distributed_r2c>)
    endif()
endif()
