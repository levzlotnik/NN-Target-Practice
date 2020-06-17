//
// Created by LevZ on 6/15/2020.
//

#ifndef BLAS_VECTORMATH_H
#define BLAS_VECTORMATH_H

#include "Vector.h"

#define VECTOR_MATH_FUNC_DECL(math_func) \
    Vector math_func(Vector v);

MACRO_MATH_FUNCTIONS(VECTOR_MATH_FUNC_DECL)

#endif //BLAS_VECTORMATH_H
