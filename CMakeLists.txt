cmake_minimum_required(VERSION 3.25.0)
project(MiOpenExample VERSION 0.1.0 LANGUAGES C CXX)

enable_language(HIP)
if(NOT DEFINED ENV{ROCM_PATH})
      set(ROCM_PATH /opt/rocm)
else()
    set(ROCM_PATH $ENV{ROCM_PATH})
endif()
list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH})
 
add_executable(${PROJECT_NAME} fusion/main.cpp fusion/miopen.hpp fusion/tensor.hpp)

add_compile_definitions(__HIP_PLATFORM_AMD__)
add_compile_definitions(__HIP_PLATFORM_HCC__)
target_include_directories(${PROJECT_NAME} PRIVATE fusion/include ${ROCM_PATH}/include)

find_package(miopen)
target_link_libraries(${PROJECT_NAME} PRIVATE MIOpen)