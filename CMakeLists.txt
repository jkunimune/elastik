cmake_minimum_required(VERSION 3.22)
project(elastik C)

set(CMAKE_C_STANDARD 99)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ../lib)

add_library(sparse SHARED src/sparse.c)
