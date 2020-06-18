//
// Created by LevZ on 6/15/2020.
//

#ifndef TARGETPRACTICE_MATRIXMATH_H
#define TARGETPRACTICE_MATRIXMATH_H

#include "Matrix.h"

#define MATRIX_MATH_FUNC_DECL(math_func) \
    Matrix math_func(Matrix m);

MACRO_MATH_FUNCTIONS(MATRIX_MATH_FUNC_DECL)

#endif //TARGETPRACTICE_MATRIXMATH_H
