cmake_minimum_required(VERSION 3.13)
project(BLAS)
set(CMAKE_CXX_STANDARD 17)

set(
        SOURCE_FILES_PROJECT_GLOBALS
        common.h
        common.cpp)

set(
        SOURCE_FILES_BLAS ${SOURCE_FILES_PROJECT_GLOBALS}
        src/Matrix.cpp include/Matrix.h
        include/common_blas.h
        src/Vector.cpp include/Vector.h
        src/VectorMath.cpp include/VectorMath.h
        src/MatrixMath.cpp include/MatrixMath.h
        include/BLAS.h src/VectorFunction.cpp include/VectorFunction.h include/Variable.cpp include/Variable.h)

add_library(blas SHARED ${SOURCE_FILES_BLAS})

set(
        SOURCE_FILES_DISTRIBUTIONS ${SOURCE_FILES_PROJECT_GLOBALS}
        include/distributions/DistributionBase.h
        src/distributions/Normal.cpp include/distributions/Normal.h
        src/distributions/Uniform.cpp include/distributions/Uniform.h
        src/distributions/MultivariateGaussian.cpp include/distributions/MultivariateGaussian.h
        src/distributions/RandomVariable.cpp include/distributions/RandomVariable.h
        src/distributions/Constant.cpp include/distributions/Constant.h
        src/distributions/MultivariateConstant.cpp include/distributions/MultivariateConstant.h
        include/distributions/distributions.h
        src/distributions/Deterministic.cpp include/distributions/Deterministic.h
)

add_library(distributions SHARED ${SOURCE_FILES_DISTRIBUTIONS})
# Add executable target with source files listed in SOURCE_FILES variable
add_executable(test_utility test.cpp)
target_link_libraries(test_utility distributions blas)