cmake_minimum_required(VERSION 3.10)
project(TargetPractice)
set(CMAKE_CXX_STANDARD 17)
#set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address")
#set (CMAKE_LINKER_FLAGS "${CMAKE_LINKER_FLAGS} -fsanitize=address")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ffloat-store")
add_subdirectory(utils)

add_subdirectory(blas)

file(GLOB
     SOURCE_FILES_DISTRIBUTIONS ${SOURCE_FILES_PROJECT_GLOBALS}
     distributions/*.h distributions/*.cpp)

# add_library(distributions SHARED ${SOURCE_FILES_DISTRIBUTIONS})
# target_link_libraries(distributions blas)

add_subdirectory(autograd)
target_include_directories(autograd PUBLIC .)

set(SOURCE_FILES_NN ${SOURCE_FILES_PROJECT_GLOBAL}
    nn/Module.cpp nn/Module.h)

add_library(nn SHARED ${SOURCE_FILES_NN})
target_link_libraries(nn autograd)

# Tests
add_subdirectory(tests)

