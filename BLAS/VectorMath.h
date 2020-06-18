//
// Created by LevZ on 6/15/2020.
//

#ifndef TARGETPRACTICE_VECTORMATH_H
#define TARGETPRACTICE_VECTORMATH_H

#include "Vector.h"
#include <unordered_map>

#define VECTOR_MATH_FUNC_DECL(math_func) \
    Vector math_func(Vector v);

MACRO_MATH_FUNCTIONS(VECTOR_MATH_FUNC_DECL)


typedef float (*elemwise_t)(float);
const unordered_map<elemwise_t, pair<string, elemwise_t>>
        elemwise_mapping
        {
            /*
            {func,       {name, derivative}}
             */
            {sin,     {"sin", [](float x){return cos(x);}} },
            {cos,     {"cos", [](float x){return -sin(x);}} },
            {tan,     {"tan", [](float x){return 1/(cos(x)*cos(x));}} },
            {sqrt,    {"sqrt", [](float x){return 0.5f/sqrt(x);}} },
            {log,     {"log", [](float x){return 1/x;}} },
            {log2,    {"log2", [](float x){const float ln2 = log(2); return 1/(x*ln2);}} },
            {log10,   {"log10", [](float x){const float ln10 = log(10); return 1/(x*ln10);}} },
            {log1p,   {"log1p", [](float x){return 1/(1+x);}} },
            {abs,     {"abs", [](float x){ return x < 0 ? -1.0f : 1.0f;}} },
            {exp,     {"exp", exp} },
            {exp2,    {"exp2", [](float x){const float ln2 = log(2); return ln2 * exp2(x);}} },
            // TODO - implement all, if needed.
        };

#endif //TARGETPRACTICE_VECTORMATH_H
