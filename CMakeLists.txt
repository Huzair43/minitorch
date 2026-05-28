cmake_minimum_required(VERSION 3.15)
project(minitorch C)

set(CMAKE_C_STANDARD 11)

# sources
file(GLOB_RECURSE SOURCES src/*.c)

# library
add_library(minitorch STATIC ${SOURCES})

# IMPORTANT: include path pour TOUTES les targets liées
target_include_directories(minitorch PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

# example
add_executable(example examples/main.c)
target_include_directories(example PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(example minitorch)

# tests
enable_testing()

if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_tensor.c")
    add_executable(test_tensor tests/test_tensor.c)
    target_include_directories(test_tensor PRIVATE 
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/tests
    )
    target_link_libraries(test_tensor minitorch m)
    add_test(NAME TensorUnitTests COMMAND test_tensor)
    
    # Make tests run with: cmake --build . --target test
    # Or: ctest
endif()