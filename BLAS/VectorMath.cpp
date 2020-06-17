//
// Created by LevZ on 6/15/2020.
//
#include "VectorMath.h"

#define VECTOR_MATH_FUNC_DEF(math_func) \
    Vector math_func(Vector v) { return v.apply([](float& x) { return math_func(x);} ); }

MACRO_MATH_FUNCTIONS(VECTOR_MATH_FUNC_DEF)
