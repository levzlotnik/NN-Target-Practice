//
// Created by LevZ on 6/15/2020.
//

#ifndef TARGETPRACTICE_VECTORMATH_H
#define TARGETPRACTICE_VECTORMATH_H

#include "Vector.h"
#include <unordered_map>
#include "../common_math.h"

#define VECTOR_MATH_FUNC_DECL(math_func) \
    Vector math_func(Vector v);

MACRO_MATH_FUNCTIONS(VECTOR_MATH_FUNC_DECL)


#endif //TARGETPRACTICE_VECTORMATH_H
