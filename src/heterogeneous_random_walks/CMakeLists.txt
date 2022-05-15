cmake_minimum_required(VERSION 3.4)
project(heterogeneous_random_walks)

add_subdirectory(pybind11)

list(APPEND CMAKE_PREFIX_PATH "/dfs/scratch1/vid/pytorch/torch")
find_package(Torch REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(CMAKE_CXX_STANDARD 14)

message(STATUS "variable ${TORCH_LIBRARIES}")

pybind11_add_module(random_walks src/main.cpp)
target_link_libraries(random_walks PUBLIC "${TORCH_LIBRARIES}" /dfs/scratch1/vid/pytorch/torch/lib/libtorch_python.so)