//
// Created by LevZ on 6/16/2020.
//

#ifndef TARGETPRACTICE_VECTORFUNCTION_H
#define TARGETPRACTICE_VECTORFUNCTION_H

#include "../BLAS/BLAS.h"
#include "Variable.h"

class Functor {
public:
    vector<int> input_shapes;
    int output_shape;
    const string name;

    Functor(const vector<int>& input_shapes, int output_shape, string name);

    virtual ~Functor() = default;

    virtual Vector operator()(const vector<Vector>& args) const = 0;

    void check_args(const vector<Vector>& args) const; // throws runtime_error if mismatch

    virtual Matrix jac(int i, const vector<Vector>& inputs, const Vector& output) const = 0;

    [[nodiscard]] virtual Functor* clone() const = 0;

    Variable* operator()(vector<Variable*>& args, bool requires_grad=true);
};

class ReduceAndGather : public Functor {
    typedef float (*reduce_t)(Vector inp);
    typedef Vector(*reducejac_t)(Vector inp, float output);
    reduce_t func;
    reducejac_t reducejac;
public:
    ReduceAndGather(const vector<int>& input_shapes, reduce_t func, reducejac_t jac, string func_name);

    Vector operator()(const vector<Vector>& args) const override;
    Matrix jac(int i, const vector<Vector>& inputs, const Vector& output) const override;

    [[nodiscard]] Functor *clone() const override;
};

class Concat : public Functor {
private:
    vector<Matrix> const_jacs;
public:
    explicit Concat(const vector<int>& input_shapes);
    Vector operator()(const vector<Vector>& args) const override;

    Matrix jac(int i, const vector<Vector>& inputs, const Vector& output) const override;

    [[nodiscard]] Functor *clone() const override;
};

class Slice : public Functor {
private:
    Matrix const_jac;
public:
    int begin;
    int end;
    int step;

    Slice(int b, int e, int input_shape, int step=1);

    Vector operator()(const vector<Vector> &args) const override;

    Matrix jac(int i, const vector<Vector> &inputs, const Vector &output) const override;

    [[nodiscard]] Functor *clone() const override;
};

class Elemwise : public Functor {
private:
    elemwise_t func;
    elemwise_t dfunc;
public:

    Elemwise(elemwise_t func, int shape);
    Elemwise(elemwise_t func, int shape, elemwise_t dfunc, string func_name);

    Vector operator()(const vector<Vector> &args) const override;
    Vector operator()(const Vector& v) const;

    Matrix jac(int i, const vector<Vector> &inputs, const Vector &output) const override;

    [[nodiscard]] Functor *clone() const override;
};

#endif //TARGETPRACTICE_VECTORFUNCTION_H
