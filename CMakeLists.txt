cmake_minimum_required(VERSION 3.18)
project(uammd LANGUAGES CXX CUDA)

# Options for user overrides
option(DOUBLE_PRECISION "Enable double precision" OFF)
option(USE_MKL "Use Intel MKL for LAPACK/BLAS" OFF)
set(VERBOSITY 0 CACHE STRING "UAMMD verbosity level")
set(MODULE_NAME "uammd" CACHE STRING "Python module name")


# CUDA
find_package(CUDA 12.0 REQUIRED)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CUDA_ARCHITECTURES 75 CACHE STRING "CUDA architectures")

# Find CUDA libraries manually for linking
find_library(CUDA_CUDART_LIBRARY cudart HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64 ${CUDA_TOOLKIT_ROOT_DIR}/lib ${CUDA_TOOLKIT_ROOT_DIR}/targets/x86_64-linux/lib)
find_library(CUDA_cufft_LIBRARY cufft HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64 ${CUDA_TOOLKIT_ROOT_DIR}/lib ${CUDA_TOOLKIT_ROOT_DIR}/targets/x86_64-linux/lib)
find_library(CUDA_cublas_LIBRARY cublas HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64 ${CUDA_TOOLKIT_ROOT_DIR}/lib ${CUDA_TOOLKIT_ROOT_DIR}/targets/x86_64-linux/lib)
if(NOT CUDA_CUDART_LIBRARY OR NOT CUDA_cufft_LIBRARY OR NOT CUDA_cublas_LIBRARY)
    message(FATAL_ERROR "Could not find required CUDA libraries (cudart, cufft, cublas)")
endif()

# Python and pybind11
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED PATHS ${CMAKE_SOURCE_DIR}/pybind11)

# LAPACK/BLAS/MKL
if(USE_MKL)
    find_package(MKL REQUIRED)
    set(LAPACK_LIBS MKL::MKL)
    add_compile_definitions(USE_MKL)
else()
    find_package(LAPACK REQUIRED)
    set(LAPACK_LIBS LAPACK::LAPACK)
endif()

# UAMMD include directories
set(UAMMD_ROOT ${CMAKE_SOURCE_DIR}/uammd)
include_directories(
    ${CUDA_INCLUDE_DIRS}
    ${UAMMD_ROOT}/src
    ${UAMMD_ROOT}/src/third_party
)

# pybind11 include
include_directories(${pybind11_INCLUDE_DIRS})

# Verbosity and precision
add_compile_definitions(MAXLOGLEVEL=${VERBOSITY})
if(DOUBLE_PRECISION)
    add_compile_definitions(DOUBLE_PRECISION)
endif()

# Sources
set(UAMMD_WRAPPER_SRC ${CMAKE_SOURCE_DIR}/uammd_wrapper.cu)
set(UAMMD_PYTHON_SRC  ${CMAKE_SOURCE_DIR}/uammd_python.cpp)
set(UAMMD_INTERFACE   ${CMAKE_SOURCE_DIR}/uammd_interface.h)
set(EXAMPLE_SRC      ${CMAKE_SOURCE_DIR}/example.cpp)


# Python module (shared library)
pybind11_add_module(${MODULE_NAME} MODULE ${UAMMD_WRAPPER_SRC} ${UAMMD_PYTHON_SRC})
target_link_libraries(${MODULE_NAME} PRIVATE ${LAPACK_LIBS} ${CUDA_CUDART_LIBRARY} ${CUDA_cufft_LIBRARY} ${CUDA_cublas_LIBRARY})
set_target_properties(${MODULE_NAME} PROPERTIES CUDA_ARCHITECTURES ${CUDA_ARCHITECTURES})

# Example executable
add_executable(example ${EXAMPLE_SRC} ${UAMMD_WRAPPER_SRC})
target_link_libraries(example PRIVATE ${LAPACK_LIBS} ${CUDA_CUDART_LIBRARY} ${CUDA_cufft_LIBRARY} ${CUDA_cublas_LIBRARY})
set_target_properties(example PROPERTIES CUDA_ARCHITECTURES ${CUDA_ARCHITECTURES})


# Install rules (optional)
install(TARGETS example DESTINATION bin)

# Install Python module directly to the current Python environment's site-packages
execute_process(
  COMMAND ${Python3_EXECUTABLE} -c "import site; print(site.getsitepackages()[0])"
  OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
install(TARGETS ${MODULE_NAME}
        LIBRARY DESTINATION ${PYTHON_SITE_PACKAGES})

# Clean target
set_directory_properties(PROPERTIES ADDITIONAL_MAKE_CLEAN_FILES "${MODULE_NAME}*.so;uammd_python.o;uammd_wrapper.o;example")
