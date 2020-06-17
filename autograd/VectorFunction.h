//
// Created by LevZ on 6/16/2020.
//

#ifndef BLAS_VECTORFUNCTION_H
#define BLAS_VECTORFUNCTION_H

#include "../BLAS/BLAS.h"
#include "Variable.h"

class VectorFunction {
public:
    vector<int> input_shapes;
    int output_shape;

    VectorFunction(const vector<int>& input_shapes, int output_shape) :
        input_shapes(input_shapes), output_shape(output_shape) {}

    virtual ~VectorFunction() = default;

    virtual Vector operator()(const vector<Vector>& args) const = 0;

    void check_args(const vector<Vector>& args) const; // throws runtime_error if mismatch

    virtual Matrix jac(int i, const vector<Vector>& inputs, const Vector& output) const = 0;

    virtual VectorFunction* clone() const = 0;

    Variable operator()(const vector<Variable>& args, bool requires_grad=true);
};

class ElementwiseFunction : public VectorFunction {
    typedef float (*elemwise_t)(Vector inp);
    typedef Vector(*elemwisejac_t)(Vector inp, float output);
    elemwise_t func;
    elemwisejac_t elemwisejac;
public:
    ElementwiseFunction(
        const vector<int>& input_shapes,
        elemwise_t func, elemwisejac_t jac
        ) :
            VectorFunction(input_shapes, input_shapes.size()), func(func), elemwisejac(jac) {}

    Vector operator()(const vector<Vector>& args) const override;
    Matrix jac(int i, const vector<Vector>& inputs, const Vector& output) const override;
};

class Concat : public VectorFunction {
public:
    Concat(const vector<int>& input_shapes) :
        VectorFunction(input_shapes, std::accumulate(input_shapes.begin(), input_shapes.end(), 0)) { }
    Vector operator()(const vector<Vector>& args) const override;
    Matrix jac(int i, const vector<Vector>& inputs, const Vector& output) const override;
};

#endif //BLAS_VECTORFUNCTION_H
