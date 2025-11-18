cmake_minimum_required(VERSION 3.24)
project(hknn_embed LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler flags
if(MSVC)
    add_compile_options(/W4 /EHsc)
    add_compile_options(/permissive-)
else()
    add_compile_options(-Wall -Wextra -pedantic)
    add_compile_options(-march=native)  # Enable SIMD optimizations
endif()

# No Boost dependencies - using STL only

# Check for AVX2/AVX-512 support
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
check_cxx_compiler_flag("-mavx512f" COMPILER_SUPPORTS_AVX512)

if(COMPILER_SUPPORTS_AVX2)
    add_compile_definitions(HKNN_USE_AVX2)
    if(COMPILER_SUPPORTS_AVX512)
        add_compile_definitions(HKNN_USE_AVX512)
    endif()
endif()

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Library sources
set(HKNN_SOURCES
    src/graph/knn_bruteforce.cpp
    src/graph/pij.cpp
    src/graph/coarsen.cpp
    src/embed/sampler.cpp
    src/embed/sharing.cpp
    src/embed/optimizer.cpp
    src/embed/hierarchy.cpp
)

# Header-only modules don't need .cpp files
set(HKNN_HEADERS
    include/hknn/io/reader.hpp
    include/hknn/io/writer.hpp
    include/hknn/graph/csr_graph.hpp
    include/hknn/graph/knn_bruteforce.hpp
    include/hknn/graph/random_projection_tree.hpp
    include/hknn/graph/nn_descent.hpp
    include/hknn/graph/knn_approximate.hpp
    include/hknn/graph/pij.hpp
    include/hknn/graph/coarsen.hpp
    include/hknn/math/kernels.hpp
    include/hknn/math/rng.hpp
    include/hknn/embed/sampler.hpp
    include/hknn/embed/gradients.hpp
    include/hknn/embed/sharing.hpp
    include/hknn/embed/optimizer.hpp
    include/hknn/embed/hierarchy.hpp
    include/hknn/cli/args.hpp
)

# Static library
add_library(hknn STATIC ${HKNN_SOURCES} ${HKNN_HEADERS})

# No external libraries needed - pure STL
target_compile_features(hknn PRIVATE cxx_std_20)
target_include_directories(hknn PUBLIC ${CMAKE_SOURCE_DIR}/include)

# Executable
add_executable(embed_hknn src/cli/main.cpp)
target_link_libraries(embed_hknn PRIVATE hknn)

# Test framework (Catch2 - header only, will download if not found)
include(FetchContent)
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.5.0
)
FetchContent_MakeAvailable(Catch2)

# Tests
set(TEST_SOURCES
    tests/test_io.cpp
    tests/test_graph.cpp
    tests/test_embed.cpp
    tests/test_integration.cpp
)

add_executable(hknn_tests ${TEST_SOURCES})
target_link_libraries(hknn_tests PRIVATE hknn Catch2::Catch2WithMain)
target_compile_features(hknn_tests PRIVATE cxx_std_20)

include(CTest)
include(Catch)
catch_discover_tests(hknn_tests)

# Installation
install(TARGETS hknn embed_hknn
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)
install(DIRECTORY include/hknn DESTINATION include)


