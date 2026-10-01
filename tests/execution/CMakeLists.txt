add_numsim_fft_test(sequential_executor_test SOURCES sequential_test.cpp)
target_include_directories(sequential_executor_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

if(NUMSIM_FFT_ENABLE_OPENMP)
    add_numsim_fft_test(openmp_executor_test SOURCES openmp_test.cpp)
    target_include_directories(openmp_executor_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
endif()

if(NUMSIM_FFT_ENABLE_HPX)
    add_numsim_fft_test(hpx_executor_test SOURCES hpx_test.cpp NO_MAIN)
    target_include_directories(hpx_executor_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
endif()
