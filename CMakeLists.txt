cmake_minimum_required(VERSION 3.13)
project(TargetPractice)
set(CMAKE_CXX_STANDARD 20)
#set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address")
#set (CMAKE_LINKER_FLAGS "${CMAKE_LINKER_FLAGS} -fsanitize=address")
add_subdirectory(utils)

set(
        SOURCE_FILES_PROJECT_GLOBALS
        common.h
        common.cpp common_math.h)

set(
        SOURCE_FILES_BLAS ${SOURCE_FILES_PROJECT_GLOBALS}
        BLAS/MatrixMath.h BLAS/MatrixMath.cpp
        BLAS/VectorMath.cpp BLAS/VectorMath.h
        BLAS/BLAS.h
        BLAS/Matrix.cpp BLAS/Matrix.h
        BLAS/Vector.cpp BLAS/Vector.h
        BLAS/common_blas.h
        BLAS/Tensor.cpp BLAS/Tensor.h)

add_library(BLAS SHARED ${SOURCE_FILES_BLAS})

file(
        GLOB
        SOURCE_FILES_DISTRIBUTIONS ${SOURCE_FILES_PROJECT_GLOBALS}
        distributions/*.h distributions/*.cpp
)
add_library(distributions SHARED ${SOURCE_FILES_DISTRIBUTIONS})
target_link_libraries(distributions BLAS)

set(
        SOURCE_FILES_AUTOGRAD ${SOURCE_FILES_PROJECT_GLOBALS}
        autograd/autograd.h
        autograd/variable/AutogradVariable.cpp autograd/variable/AutogradVariable.h
        autograd/variable/VariableBase.h autograd/variable/VariableBase.cpp
        autograd/Functor.cpp autograd/Functor.h
        autograd/variable/RandomVariable.cpp autograd/variable/RandomVariable.h
        autograd/variable/Parameter.cpp autograd/variable/Parameter.h
        autograd/variable/InputBuffer.cpp autograd/variable/InputBuffer.h
        autograd/Loss.cpp autograd/Loss.h autograd/variable/Constant.h autograd/VariableMath.cpp autograd/VariableMath.h)

add_library(autograd SHARED ${SOURCE_FILES_AUTOGRAD})
target_link_libraries(autograd distributions graph2dot)

set(
        SOURCE_FILES_NN ${SOURCE_FILES_PROJECT_GLOBAL}
        nn/Module.cpp nn/Module.h
)
add_library(nn SHARED ${SOURCE_FILES_NN})
target_link_libraries(nn autograd)

# Tests
add_executable(test_BLAS tests/test_tensor.cpp)
target_link_libraries(test_BLAS BLAS)

add_executable(test_distributions tests/test_distributions.cpp)
target_link_libraries(test_distributions distributions)

add_executable(test_autograd tests/test_autograd.cpp)
target_link_libraries(test_autograd autograd)
