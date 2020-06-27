//
// Created by LevZ on 6/27/2020.
//

#include "VariableMath.h"
#include "Functor.h"
#include "Loss.h"
#include "variable/Constant.h"

Variable operator+(const Variable &v1, const Variable &v2) {
    if (v1->shape() == 1) {
        ScalarElemwise functor("add", v2->shape(), true);
        return functor(v1, v2);
    }
    else if (v2->shape() == 1){
        ScalarElemwise functor("add", v1->shape(), false);
        return functor(v1, v2);
    }
    BinaryElemwise functor{"add", v1->shape()};
    return functor(v1, v2);

}

Variable operator*(const Variable &v1, const Variable &v2) {
    if (v1->shape() == 1) {
        ScalarElemwise functor("mul", v2->shape(), true);
        return functor(v1, v2);
    }
    else if (v2->shape() == 1){
        ScalarElemwise functor("mul", v1->shape(), false);
        return functor(v1, v2);
    }
    BinaryElemwise functor{"mul", v1->shape()};
    return functor(v1, v2);
}

Variable operator/(const Variable &v1, const Variable &v2) {
    if (v1->shape() == 1) {
        ScalarElemwise functor("div", v2->shape(), true);
        return functor(v1, v2);
    }
    else if (v2->shape() == 1){
        ScalarElemwise functor("div", v1->shape(), false);
        return functor(v1, v2);
    }
    BinaryElemwise functor{"div", v1->shape()};
    return functor(v1, v2);
}

Variable operator-(const Variable &v1, const Variable &v2) {
    if (v1->shape() == 1) {
        ScalarElemwise functor("sub", v2->shape(), true);
        return functor(v1, v2);
    }
    else if (v2->shape() == 1){
        ScalarElemwise functor("sub", v1->shape(), false);
        return functor(v1, v2);
    }
    BinaryElemwise functor{"sub", v1->shape()};
    return functor(v1, v2);
}

Variable operator+(float scalar, const Variable &v) {
    ScalarElemwise functor("add", v->shape(), true);
    auto scalar_const = Constant::make("constant(" + to_string(scalar) + ")", {scalar});
    return functor(scalar_const, v);
}

Variable operator*(float scalar, const Variable &v) {
    ScalarElemwise functor("mul", v->shape(), true);
    auto scalar_const = Constant::make("constant(" + to_string(scalar) + ")", {scalar});
    return functor(scalar_const, v);
}

Variable operator/(float scalar, const Variable &v) {
    ScalarElemwise functor("div", v->shape(), true);
    auto scalar_const = Constant::make("constant(" + to_string(scalar) + ")", {scalar});
    return functor(scalar_const, v);
}

Variable operator-(float scalar, const Variable &v) {
    ScalarElemwise functor("sub", v->shape(), true);
    auto scalar_const = Constant::make("constant(" + to_string(scalar) + ")", {scalar});
    return functor(scalar_const, v);
}

Variable operator+(const Variable &v, float scalar) {
    ScalarElemwise functor("add", v->shape(), false);
    auto scalar_const = Constant::make("constant(" + to_string(scalar) + ")", {scalar});
    return functor(v, scalar_const);
}

Variable operator*(const Variable &v, float scalar) {
    ScalarElemwise functor("mul", v->shape(), false);
    auto scalar_const = Constant::make("constant(" + to_string(scalar) + ")", {scalar});
    return functor(v, scalar_const);
}

Variable operator/(const Variable &v, float scalar) {
    ScalarElemwise functor("div", v->shape(), false);
    auto scalar_const = Constant::make("constant(" + to_string(scalar) + ")", {scalar});
    return functor(v, scalar_const);
}

Variable operator-(const Variable &v, float scalar) {
    ScalarElemwise functor("sub", v->shape(), false);
    auto scalar_const = Constant::make("constant(" + to_string(scalar) + ")", {scalar});
    return functor(v, scalar_const);
}

static float binary_op_pow(float& b, float& e){
    return pow(b, e);
}

static float dpow_dbase(float b, float e, float o) {
    if (e == 0 || b == 0)
        return 0;
    return e*(o / b);
}

static float dpow_dexp(float b, float e, float o){
    if (b == 0)
        return 0;
    return o * log(b);
}

Variable pow(const Variable &v1, const Variable &v2) {
    BinaryElemwise functor(binary_op_pow, {dpow_dbase, dpow_dexp}, v1->shape(), "pow");
    return functor(v1, v2);
}

Variable pow(const Variable &v, float x) {
    ScalarElemwise functor(binary_op_pow, {dpow_dbase, dpow_dexp}, v->shape(), "pow", false);
    auto scalar_const = Constant::make("constant(" + to_string(x) + ")", {x});
    return functor(v, scalar_const);
}

Variable pow(float x, const Variable &v) {
    ScalarElemwise functor(binary_op_pow, {dpow_dbase, dpow_dexp}, v->shape(), "pow", true);
    auto scalar_const = Constant::make("constant(" + to_string(x) + ")", {x});
    return functor(scalar_const, v);
}

Variable sum(const Variable &v) {
    Reduce functor(v->shape(),
            [](Vector v){return v.sum();},
            [](Vector v, float out){ const auto res = Vector::ones_like(v); return res;},
            "sum");
    return functor(v);
}

Variable mean(const Variable &v) {
    Reduce functor(v->shape(),
                   [](Vector v){return v.mean();},
                   [](Vector v, float out){return Vector::ones_like(v) / v.n;},
                   "mean");
    return functor(v);
}

Variable mse(const Variable &v_true, const Variable &v_pred) {
    MSELoss criterion{v_true->shape()};
    return criterion(v_true, v_pred);
}

#define DEF_MATH_FUNCTION_VARIABLE(func) \
    Variable func(const Variable& v) { \
       return Elemwise(#func, v->shape())(v); \
    }

MACRO_MATH_FUNCTIONS(DEF_MATH_FUNCTION_VARIABLE)