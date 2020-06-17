//
// Created by LevZ on 6/15/2020.
//
#include "MatrixMath.h"

#define MATRIX_MATH_FUNC_DEF(math_func) \
    Matrix math_func(Matrix v) { return v.apply([](float& x) { return math_func(x);} ); }

MACRO_MATH_FUNCTIONS(MATRIX_MATH_FUNC_DEF)
