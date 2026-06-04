cmake_minimum_required(VERSION 3.15)
project(minitorch C)
set(CMAKE_C_STANDARD 11)

# ── Sources ────────────────────────────────────────────
file(GLOB_RECURSE SOURCES src/*.c)

# ── Bibliothèque statique ──────────────────────────────
add_library(minitorch STATIC ${SOURCES})
target_include_directories(minitorch PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

# ── Executables examples ───────────────────────────────
add_executable(example examples/main.c)
target_include_directories(example PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(example minitorch m)

add_executable(demo_train examples/demo_train.c)
target_include_directories(demo_train PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(demo_train minitorch m)

add_executable(demo_xor examples/demo_xor.c)
target_include_directories(demo_xor PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(demo_xor minitorch m)

add_executable(demo_multiclass examples/demo_multiclass.c)
target_include_directories(demo_multiclass PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(demo_multiclass minitorch m)

# ── Tests ──────────────────────────────────────────────
enable_testing()

if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_tensor.c")
    add_executable(test_tensor tests/test_tensor.c)
    target_include_directories(test_tensor PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/tests
    )
    target_link_libraries(test_tensor minitorch m)
    add_test(NAME TensorUnitTests COMMAND test_tensor)
endif()

if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_autograd.c")
    add_executable(test_autograd tests/test_autograd.c)
    target_include_directories(test_autograd PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/tests
    )
    target_link_libraries(test_autograd minitorch m)
    add_test(NAME AutogradUnitTests COMMAND test_autograd)
endif()

if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_nn.c")
    add_executable(test_nn tests/test_nn.c)
    target_include_directories(test_nn PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/tests
    )
    target_link_libraries(test_nn minitorch m)
    add_test(NAME NNUnitTests COMMAND test_nn)
endif()

if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_optim.c")
    add_executable(test_optim tests/test_optim.c)
    target_include_directories(test_optim PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/tests
    )
    target_link_libraries(test_optim minitorch m)
    add_test(NAME OptimUnitTests COMMAND test_optim)
endif()

if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_data.c")
    add_executable(test_data tests/test_data.c)
    target_include_directories(test_data PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/tests
    )
    target_link_libraries(test_data minitorch m)
    add_test(NAME DataUnitTests COMMAND test_data)
endif()

if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_xor.c")
    add_executable(test_xor tests/test_xor.c)
    target_include_directories(test_xor PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/tests
    )
    target_link_libraries(test_xor minitorch m)
    add_test(NAME XORTrainingTests COMMAND test_xor)
endif()

if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_serialization.c")
    add_executable(test_serialization tests/test_serialization.c)
    target_include_directories(test_serialization PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/tests
    )
    target_link_libraries(test_serialization minitorch m)
    add_test(NAME SerializationUnitTests COMMAND test_serialization)
endif()

if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_multiclass.c")
    add_executable(test_multiclass tests/test_multiclass.c)
    target_include_directories(test_multiclass PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/tests
    )
    target_link_libraries(test_multiclass minitorch m)
    add_test(NAME MulticlassUnitTests COMMAND test_multiclass)
endif()
