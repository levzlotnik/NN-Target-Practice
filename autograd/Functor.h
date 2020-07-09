//
// Created by LevZ on 6/16/2020.
//

#ifndef TARGETPRACTICE_FUNCTOR_H
#define TARGETPRACTICE_FUNCTOR_H

#include "../BLAS/BLAS.h"
#include "variable/VariableBase.h"
#include <array>

class Functor {
public:
    vector<int> input_shapes;
    int output_shape;
    const string name;

    Functor(const vector<int>& input_shapes, int output_shape, string name);

    virtual ~Functor() = default;

    virtual Vector operator()(const vector<Vector>& args) const = 0;

    void check_args(const vector<Vector>& args) const; // throws runtime_error if mismatch
    void check_args(const vector<Variable>& args) const;

    virtual Matrix jac(int i, const vector<Vector>& inputs, const Vector& output) const = 0;

    [[nodiscard]] virtual Functor* clone() const = 0;

    Variable operator()(const vector<Variable>& args, bool requires_grad=true) const;
};

class ReduceAndGather : public Functor {
protected:
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

class Reduce : public ReduceAndGather {
protected:
    using ReduceAndGather::reduce_t;
    using ReduceAndGather::reducejac_t;
public:
    Reduce(int shape, reduce_t func, reducejac_t jac, string func_name) :
        ReduceAndGather({shape}, func, jac, func_name){}

    Vector operator()(const Vector& v) const {
        return ReduceAndGather::operator()({v});
    }

    Variable operator()(const Variable& v, bool requires_grad=true) {
        return Functor::operator()({v}, requires_grad);
    }
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
    unary_elemwise_t func;
    unary_elemwise_t dfunc;
public:

    Elemwise(string func_name, int shape);
    Elemwise(unary_elemwise_t func, unary_elemwise_t dfunc, int shape, const string &func_name);

    Vector operator()(const vector<Vector> &args) const override;
    Variable operator()(const Variable& var, bool requires_grad=true) const ;

    Matrix jac(int i, const vector<Vector> &inputs, const Vector &output) const override;

    [[nodiscard]] Functor *clone() const override;
};

class ScalarElemwise : public Functor {
private:
    binary_elemwise_t op;
    array<jac_binary_elemwise_t, 2> dx_ops;
    bool scalar_first;
    BinaryOperation op_for_vector;

public:
    ScalarElemwise(string op_name, int shape, bool scalar_first);
    ScalarElemwise(binary_elemwise_t op, pair<jac_binary_elemwise_t, jac_binary_elemwise_t> d_op, int shape,
                   const string &op_name, bool scalar_first);

    Vector operator()(const vector<Vector> &args) const override;
    Variable operator()(const Variable& v1, const Variable& v2, bool requires_grad=true) const;

    Matrix jac(int i, const vector<Vector> &inputs, const Vector &output) const override;

    Functor *clone() const override;
};


class BinaryElemwise : public Functor {
private:
    binary_elemwise_t op;
    array<jac_binary_elemwise_t, 2> dx_op;
public:
    BinaryElemwise(string op_name, int shape);
    BinaryElemwise(binary_elemwise_t op, pair<jac_binary_elemwise_t, jac_binary_elemwise_t> d_op,
                   int shape, const string &op_name);

    Vector operator()(const vector<Vector> &args) const override;
    Vector operator()(const Vector& x1, const Vector& x2) const;
    Variable operator()(const Variable& v1, const Variable& v2, bool requires_grad=true) const;

    Matrix jac(int i, const vector<Vector> &inputs, const Vector &output) const override;

    Functor *clone() const override;
};




#endif //TARGETPRACTICE_FUNCTOR_H
