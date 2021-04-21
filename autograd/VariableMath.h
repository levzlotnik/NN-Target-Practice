//
// Created by LevZ on 9/26/2020.
//

#ifndef TARGETPRACTICE_VARIABLEMATH_H
#define TARGETPRACTICE_VARIABLEMATH_H

#include <numeric>
#include "AutogradVariable.h"
#include "Functor.h"
#include "../common_math.h"

namespace autograd {

#define VARIABLE_BINARY_MATH_MACRO(op, op_name, modifier)                                               \
    template<typename T>                                                                                \
    Variable<T> modifier op(const Variable<T>& v1, const Variable<T>& v2) {                             \
        TensorTensorElemwiseFunctor<T> functor{v1.shape(), v2.shape(), op_name};                        \
        return functor({v1, v2});                                                                       \
    }                                                                                                   \
    template<typename T>                                                                                \
    Variable<T> modifier op(T scalar, const Variable<T>& v) {                                           \
        ScalarTensorElemwiseFunctor<T> functor{v.shape(), scalar, op_name, true};                       \
        return functor({v});                                                                            \
    }                                                                                                   \
    template<typename T>                                                                                \
    Variable<T> modifier op(const Variable<T>& v, T scalar) {                                           \
        ScalarTensorElemwiseFunctor<T> functor{v.shape(), scalar, op_name, false};                      \
        return functor({v});                                                                            \
    }

#define VARIABLE_MATH_OP(op, op_name) VARIABLE_BINARY_MATH_MACRO(op, op_name, operator)

    MACRO_BASIC_ARITHMETIC_OPERATORS_NAMED(VARIABLE_MATH_OP)

#define EMPTY
    VARIABLE_BINARY_MATH_MACRO(pow, "pow", EMPTY)
#undef EMPTY

#define VARIABLE_MATH_FUNC(func) \
    template<typename T>         \
    Variable<T> func(const Variable<T>& v) { \
        MathFunctor<T> functor{v.shape(), #func}; \
        return functor({v}); \
    }
    MACRO_MATH_FUNCTIONS(VARIABLE_MATH_FUNC)

    static inline string generate_op_name(){
        static int i = 0;
        using std::to_string;
        return "op__" + to_string(i++);
    }

    template<typename T>
    inline Variable<T> reduce(const Variable<T>& v,
                       const binary_op<T>& op, const binary_op<T>& op_jac, const vector<int>& dims, string op_name = "") {
        if (op_name.empty())
            op_name = generate_op_name();
        ReduceFunctor<T> functor(v.shape(), op_name, dims, op, op_jac);
        return functor({v});
    }

    template<typename T>
    inline Variable<T> reduce(const Variable<T>& v,
                       const binary_op<T>& op, const binary_op<T>& op_jac, const string& op_name = "") {
        vector<int> dims(v.shape().size());
        std::iota(dims.begin(), dims.end(), 0);
        return reduce(v, op, op_jac, dims, op_name);
    }

    template<typename T>
    inline Variable<T> reduce(const Variable<T>& v,
                       const binary_op<T>& op, const binary_op<T>& op_jac, int dim, const string& op_name = "") {
        vector<int> dims = {dim};
        return reduce(v, op, op_jac, dims, op_name);
    }

    template<typename T>
    inline Variable<T> sum(const Variable<T>& v) {
        ReduceFunctor<T> functor{v.shape(), "add"};
        return functor({v});
    }

    template<typename T>
    inline Variable<T> sum(const Variable<T>& v, const vector<int>& dims) {
        ReduceFunctor<T> functor{v.shape(), "add", dims};
        return functor({v});
    } 

    template<typename T>
    inline Variable<T> sum(const Variable<T>& v, int dim){
        return sum(v, vector<int>{dim});
    }

    template<typename T>
    inline Variable<T> matmul(const Variable<T>& v1, const Variable<T>& v2) {
        MatMulFunctor<T> functor{v1.shape(), v2.shape()};
        return functor(v1, v2);
    }
}

#endif //TARGETPRACTICE_VARIABLEMATH_H
