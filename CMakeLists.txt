# CMake configuration file for matrix_multip_kokkos

cmake_minimum_required(VERSION 3.10)
project(matrix_multip_kokkos LANGUAGES CXX CUDA)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set paths for Kokkos and CUDA installations (adjust these paths as necessary)
set(KOKKOS_INSTALL_PATH /home/hice1/lnarayanan7/kokkos-install)
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/pace-apps/spack/packages/linux-rhel9-x86_64_v3/gcc-12.3.0/cuda-12.1.1-tftm224bafcysdbw3y6ufurixpcancaq)

# Set the CXX compiler to nvcc_wrapper provided by Kokkos
set(CMAKE_CXX_COMPILER ${KOKKOS_INSTALL_PATH}/bin/nvcc_wrapper)
set(CMAKE_CUDA_COMPILER ${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc)

# Set CUDA architectures
set(CMAKE_CUDA_ARCHITECTURES 90) # Set appropriate CUDA architectures for your GPUs

# Find the Kokkos package
find_package(Kokkos REQUIRED PATHS ${KOKKOS_INSTALL_PATH})

# Include directories for Kokkos
include_directories(${Kokkos_INCLUDE_DIRS})

# Add the source file
set(SOURCES matrix_multip_kokkos.cpp)

# Create the executable
add_executable(matrix_multip_kokkos ${SOURCES})

# Link the executable with Kokkos and CUDA libraries
target_link_libraries(matrix_multip_kokkos Kokkos::kokkos cuda cudart)

# Set compile options for CUDA
target_compile_options(matrix_multip_kokkos PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda -arch=sm_70>
)

# Enable OpenMP for multi-threading
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    target_link_libraries(matrix_multip_kokkos OpenMP::OpenMP_CXX)
    target_compile_options(matrix_multip_kokkos PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-fopenmp>)
endif()

# Display message indicating successful configuration
message(STATUS "CMake configuration completed for multi-GPU matrix multiplication.")
