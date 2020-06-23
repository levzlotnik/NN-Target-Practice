cmake_minimum_required(VERSION 3.13)
project(TargetPractice)
set(CMAKE_CXX_STANDARD 17)

set(
        SOURCE_FILES_PROJECT_GLOBALS
        common.h
        common.cpp)

set(
        SOURCE_FILES_BLAS ${SOURCE_FILES_PROJECT_GLOBALS}
        BLAS/MatrixMath.h BLAS/MatrixMath.cpp
        BLAS/VectorMath.cpp BLAS/VectorMath.h
        BLAS/BLAS.h
        BLAS/Matrix.cpp BLAS/Matrix.h
        BLAS/Vector.cpp BLAS/Vector.h
        BLAS/common_blas.h
)

add_library(BLAS SHARED ${SOURCE_FILES_BLAS})

file(
        GLOB
        SOURCE_FILES_DISTRIBUTIONS ${SOURCE_FILES_PROJECT_GLOBALS}
        distributions/*.h distributions/*.cpp
)
add_library(distributions SHARED ${SOURCE_FILES_DISTRIBUTIONS})

set(
        SOURCE_FILES_AUTOGRAD ${SOURCE_FILES_PROJECT_GLOBALS}
        autograd/AutogradVariable.cpp autograd/AutogradVariable.h
        autograd/Variable.h autograd/Variable.cpp
        autograd/Functor.cpp autograd/Functor.h
        autograd/RandomVariable.cpp autograd/RandomVariable.h
)
add_library(autograd SHARED ${SOURCE_FILES_AUTOGRAD})

# Tests
add_executable(test_BLAS tests/test_BLAS.cpp)
target_link_libraries(test_BLAS BLAS)

add_executable(test_distributions tests/test_distributions.cpp)
target_link_libraries(test_distributions BLAS distributions)

add_executable(test_autograd tests/test_autograd.cpp)
target_link_libraries(test_autograd distributions BLAS autograd)
