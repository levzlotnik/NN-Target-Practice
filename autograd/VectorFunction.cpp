//
// Created by LevZ on 6/16/2020.
//

#include "VectorFunction.h"

Vector ElementwiseFunction::operator()(const vector<Vector>& args) const {
    check_args(args);
    Vector res(output_shape);
    for (int i = 0; i < output_shape; ++i)
        res[i] = this->func(args[i]);
    return res;
}

Matrix ElementwiseFunction::jac(int i, const vector<Vector>& inputs, const Vector& output) const {
    int inp_shape = input_shapes[i];
    Matrix res(output_shape, inp_shape, 0);
    res.set_row(i, elemwisejac(inputs[i], output[i]));
    return res;
}


Vector Concat::operator()(const vector<Vector>& args) const {
    check_args(args);
    return Vector::concat(args);
}

Matrix Concat::jac(int i, const vector<Vector>& inputs, const Vector& output) const {
    return VectorFunction::jac(i, inputs, output);
}

void VectorFunction::check_args(const vector<Vector>& args) const {
    if (args.size() != input_shapes.size())
        throw runtime_error("Arguments count mismatch: " + to_string(args.size()) +
            ", " + to_string(input_shapes.size()));
    for (int i=0; i<args.size(); ++i)
        if (args[i].n != input_shapes[i])
            throw runtime_error("Shape Mismatch on argument " + to_string(i+1) + ": "
                + to_string(args[i].n) + ", " + to_string(input_shapes[i]));
}

Variable VectorFunction::operator()(const vector<Variable>& args, bool requires_grad) {

}
