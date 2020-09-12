//
// Created by LevZ on 6/27/2020.
//

#ifndef TARGETPRACTICE_VARIABLEMATH_H
#define TARGETPRACTICE_VARIABLEMATH_H

#include "VariableBase.h"
#include "../common.h"

Variable operator+(const Variable& v1, const Variable& v2);
Variable operator*(const Variable& v1, const Variable& v2);
Variable operator/(const Variable& v1, const Variable& v2);
Variable operator-(const Variable& v1, const Variable& v2);

Variable operator+(float scalar, const Variable& v);
Variable operator*(float scalar, const Variable& v);
Variable operator/(float scalar, const Variable& v);
Variable operator-(float scalar, const Variable& v);

Variable operator+(const Variable& v, float scalar);
Variable operator*(const Variable& v, float scalar);
Variable operator/(const Variable& v, float scalar);
Variable operator-(const Variable& v, float scalar);


#define DECL_MATH_FUNCTION_VARIABLE(func) \
    Variable func(const Variable& v);

MACRO_MATH_FUNCTIONS(DECL_MATH_FUNCTION_VARIABLE)

Variable pow(const Variable& v1, const Variable& v2);
Variable pow(const Variable& v, float x);
Variable pow(float x, const Variable& v);

Variable sum(const Variable& v);
Variable mean(const Variable& v);

Variable mse(const Variable& v_true, const Variable& v_pred);


#endif //TARGETPRACTICE_VARIABLEMATH_H
