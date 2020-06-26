//
// Created by LevZ on 6/26/2020.
//

#include "Loss.h"

Vector MSELoss::operator()(const vector<Vector> &args) const {
    check_args(args);
    return { 0.5f * (args[0] - args[1]).pow_(2).mean() };
}

Matrix MSELoss::jac(int i, const vector<Vector> &inputs, const Vector &output) const {
    check_args(inputs);
    auto delta = inputs[0] - inputs[1];
    delta /= (float)delta.shape();
    // return the delta on the diagonal, but respecting the position of the argument :
    // delta for first argument, -delta for the second argument
    if (i == 1) delta *= -1.0f;
    return reshape(delta,1, delta.n);
}

Functor *MSELoss::clone() const {
    return new MSELoss(*this);
}

Vector MSELoss::operator()(const Vector &val_true, const Vector &val_pred) const {
    return operator()({val_true, val_pred});
}

shared_ptr<Variable> MSELoss::operator()(const shared_ptr<Variable> &var_true, const shared_ptr<Variable> &var_pred,
                                         bool requires_grad) const {
    return Functor::operator()({var_true, var_pred}, requires_grad);
}
