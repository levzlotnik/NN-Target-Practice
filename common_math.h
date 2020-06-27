//
// Created by LevZ on 6/27/2020.
//

#ifndef TARGETPRACTICE_COMMON_MATH_H
#define TARGETPRACTICE_COMMON_MATH_H

#include <string>
#include <unordered_map>
#include <cmath>

using namespace std;
static inline float relu(float x) { return x > 0 ? x : 0; }

// Apply a macro to all math functions.
//   Note: Only add a macro function here if you can add it to the
//         unary_elemwise_map!
#define MACRO_MATH_FUNCTIONS(macro) \
    macro(sin) \
    macro(cos) \
    macro(tan) \
    macro(sqrt) \
    macro(log) \
    macro(log2) \
    macro(log10) \
    macro(log1p) \
    macro(abs) \
    macro(exp2) \
    macro(floor) \
    macro(ceil) \
    macro(round) \
    macro(exp) \
    macro(relu)


typedef float (*unary_elemwise_t)(float);

const unordered_map<string, pair<unary_elemwise_t, unary_elemwise_t>>
        unary_elemwise_mapping
        {
             /* {func,       {name, derivative}} */
                {"sin",     {sin, [](float x){return cos(x);}} },
                {"cos",     {cos, [](float x){return -sin(x);}} },
                {"tan",     {tan, [](float x){return 1/(cos(x)*cos(x));}} },
                {"sqrt",    {sqrt, [](float x){return 0.5f/sqrt(x);}} },
                {"log",     {log, [](float x){return 1/x;}} },
                {"log2",    {log2, [](float x){const float ln2 = log(2); return 1/(x*ln2);}} },
                {"log10",   {log10, [](float x){const float ln10 = log(10); return 1/(x*ln10);}} },
                {"log1p",   {log1p, [](float x){return 1/(1+x);}} },
                {"abs",     {abs, [](float x){ return x < 0 ? -1.0f : 1.0f;}} },
                {"exp",     {exp, exp} },
                {"exp2",    {exp2, [](float x){const float ln2 = log(2); return ln2 * exp2(x);}} },
                {"relu",    {relu, [](float x){return x > 0 ? 1.0f : 0 ;}} }
                // TODO - implement more, if needed.
        };

typedef float (*binary_elemwise_t)(float&, float&);
typedef float (*jac_binary_elemwise_t)(float in1, float in2, float out);

static inline float add(float& x, float& y) { return x + y; };
static inline float mul(float& x, float& y) { return x * y; };
static inline float sub(float& x, float& y) { return x - y; };
static inline float div(float& x, float& y) { return x / y; };

const unordered_map<std::string,
                    tuple<binary_elemwise_t, jac_binary_elemwise_t , jac_binary_elemwise_t>>
        binary_elemwise_mapping =
        {
                {"add",{add, [](float i1, float i2, float o){return 1.0f;}, [](float i1, float i2, float o){return 1.0f;}}},
                {"mul",{mul, [](float i1, float i2, float o){return i2;}, [](float i1, float i2, float o){return i1;}}},
                {"sub",{sub, [](float i1, float i2, float o){return 1.0f;}, [](float i1, float i2, float o){return -1.0f;}}},
                {"div",{div, [](float i1, float i2, float o){return 1/i2;}, [](float i1, float i2, float o){return -o/i2;}}}
        };

#endif //TARGETPRACTICE_COMMON_MATH_H
