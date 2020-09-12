set(SOURCE_FILES_PROJECT_GLOBALS
        ../common.h
        ../common.cpp ../common_math.h)

set(SOURCE_FILES_AUTOGRAD ${SOURCE_FILES_PROJECT_GLOBALS}
    autograd.h
    AutogradVariable.cpp AutogradVariable.h
    VariableBase.h VariableBase.cpp
    Functor.cpp Functor.h
    RandomVariable.cpp RandomVariable.h
    Parameter.cpp Parameter.h
    InputBuffer.cpp InputBuffer.h
    Loss.cpp Loss.h Constant.h VariableMath.cpp VariableMath.h Functor.cpp Functor.h)

add_library(autograd SHARED ${SOURCE_FILES_AUTOGRAD})
target_link_libraries(autograd blas)