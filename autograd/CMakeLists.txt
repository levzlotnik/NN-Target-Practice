set(SOURCE_FILES_PROJECT_GLOBALS
        ../common.h
        ../common.cpp ../common_math.h)

set(SOURCE_FILES_AUTOGRAD ${SOURCE_FILES_PROJECT_GLOBALS}
    autograd.h
    AutogradVariable.cpp AutogradVariable.h
    VariableBase.h VariableBase.cpp
    Functor.cpp Functor.h
    Parameter.h
    InputBuffer.h
    Constant.h
    Functor.cpp Functor.h
    VariableMath.h VariableMath.cpp
    Loss.cpp Loss.h)

add_library(autograd SHARED ${SOURCE_FILES_AUTOGRAD})
target_link_libraries(autograd blas graph2dot)