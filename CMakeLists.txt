cmake_minimum_required(VERSION 3.13)
project(BLAS)
set(CMAKE_CXX_STANDARD 17)

set(
        SOURCE_FILES_PROJECT_GLOBALS
        common.h
        common.cpp)

file(
        GLOB
        SOURCE_FILES_BLAS ${SOURCE_FILES_PROJECT_GLOBALS}
        BLAS/*.h BLAS/*.cpp
)

add_library(blas SHARED ${SOURCE_FILES_BLAS})

file(
        GLOB
        SOURCE_FILES_DISTRIBUTIONS ${SOURCE_FILES_PROJECT_GLOBALS}
        distributions/*.h distributions/*.cpp
)
add_library(distributions SHARED ${SOURCE_FILES_DISTRIBUTIONS})

file(
        GLOB
        SOURCE_FILES_AUTOGRAD ${SOURCE_FILES_PROJECT_GLOBALS}
        autograd/*.h autograd/*.cpp
)
add_library(autograd SHARED ${SOURCE_FILES_AUTOGRAD})

# Add executable target with source files listed in SOURCE_FILES variable
add_executable(test_utility test.cpp)
target_link_libraries(test_utility distributions blas autograd)