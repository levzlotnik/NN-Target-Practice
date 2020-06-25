//
// Created by LevZ on 6/16/2020.
//

#include "Functor.h"
#include "AutogradVariable.h"

Vector ReduceAndGather::operator()(const vector<Vector>& args) const {
    check_args(args);
    Vector res(output_shape);
    for (int i = 0; i < output_shape; ++i)
        res[i] = this->func(args[i]);
    return res;
}

Matrix ReduceAndGather::jac(int i, const vector<Vector>& inputs, const Vector& output) const {
    int inp_shape = input_shapes[i];
    Matrix res(output_shape, inp_shape, true);
    res.set_row(i, reducejac(inputs[i], output[i]));
    return res;
}

Functor *ReduceAndGather::clone() const {
    return new ReduceAndGather(*this);
}

ReduceAndGather::ReduceAndGather(const vector<int> &input_shapes,
        ReduceAndGather::reduce_t func, ReduceAndGather::reducejac_t jac, string func_name) :
    func(func), reducejac(jac),
    Functor(input_shapes,
            std::accumulate(input_shapes.begin(), input_shapes.end(), 0),
            "ReduceAndGather[" + func_name + "]")
{

}


Vector Concat::operator()(const vector<Vector>& args) const {
    check_args(args);
    return Vector::concat(args);
}

Matrix Concat::jac(int i, const vector<Vector>& inputs, const Vector& output) const {
    return const_jacs[i];
}

Functor *Concat::clone() const {
    return new Concat(*this);
}

Concat::Concat(const vector<int> &input_shapes)  :
        Functor(input_shapes,
                std::accumulate(input_shapes.begin(), input_shapes.end(), 0),
                "Concat")
        {
    int idx = 0;
    for (auto shape: input_shapes){
        Matrix curr_jac(output_shape, shape, true);
        // Make an identity matrix at the relevant indices
        for (int i=0; i < shape; ++i)
            curr_jac(idx + i, i) = 1;

        const_jacs.emplace_back(curr_jac);
        idx += shape;
    }
}

void Functor::check_args(const vector<Vector>& args) const {
    if (args.size() != input_shapes.size())
        throw runtime_error("Arguments count mismatch: " + to_string(args.size()) +
            ", " + to_string(input_shapes.size()));
    for (int i=0; i<args.size(); ++i)
        if (args[i].n != input_shapes[i])
            throw runtime_error("Shape Mismatch on argument " + to_string(i+1) + ": "
                + to_string(args[i].n) + ", " + to_string(input_shapes[i]));
}

Variable* Functor::operator()(vector<Variable*>& args, bool requires_grad) {
    string name_var = name + ".Result";
    auto res = new AutogradVariable(name, *this, requires_grad);
    for (auto arg: args)
        res->add_dependency(arg);
    res->forward();
    return res;
}

Functor::Functor(const vector<int> &input_shapes, int output_shape, string name) :
        input_shapes(input_shapes), output_shape(output_shape), name(name) {
    if (output_shape < 1 ||
        std::any_of(input_shapes.begin(), input_shapes.end(), [](int x) {return x < 1;}))
        throw runtime_error("Shapes must be positive.");
}

Vector Slice::operator()(const vector<Vector> &args) const {
    check_args(args);
    return args[0].slice(begin, end, step);
}

Matrix Slice::jac(int i, const vector<Vector> &inputs, const Vector &output) const {
    if (i != 0)
        throw out_of_range("Only 1 element available in slice.");
    return const_jac;
}

Functor *Slice::clone() const {
    return new Slice(*this);
}

Slice::Slice(int b, int e, int input_shape, int step) :
    Functor(vector<int>(1, input_shape), 1,
            "Slice(" + to_string(b) + ", " + to_string(e) + ", " + to_string(step) + ")") {
    b = normalize_index(b, input_shape);
    e = normalize_index(e, input_shape);
    output_shape = (e - b) / step;
    if (output_shape == 0)
        throw runtime_error("The slice size cannot be 0.");
    if (output_shape < 0)
        throw runtime_error("The slice and step directions contradict each other: slice direction =" +
                            to_string(e-b) + ", step = " + to_string(step));
    begin = b;
    end = e;
    this->step = step;
    const_jac = Matrix(output_shape, input_shape, true);
    for(int i=0; i<output_shape; ++i)
        const_jac(i, b + i*step) = 1;
}

Elemwise::Elemwise(elemwise_t func, int shape) :
    Elemwise(func, shape,
            elemwise_mapping.at(func).second,
            elemwise_mapping.at(func).first)
{

}

Elemwise::Elemwise(elemwise_t func, int shape, elemwise_t dfunc, string func_name) :
    func(func), dfunc(dfunc),
    Functor(vector<int>(1, shape), shape,
            "Elemwise[" + func_name + "]")
{

}

Vector Elemwise::operator()(const vector<Vector> &args) const {
    check_args(args);
    return args[0].apply(func);
}

Matrix Elemwise::jac(int i, const vector<Vector> &inputs, const Vector &output) const {
    check_args(inputs);
    if (i==0)
        throw out_of_range("Elemwise only accepts a single vector");
    Matrix res(output_shape, output_shape, true);
    res.set_diag(inputs[0].apply(dfunc));
    return res;
}

Functor *Elemwise::clone() const {
    return new Elemwise(*this);
}

Vector Elemwise::operator()(const Vector& v) const {
    return this->operator()(vector<Vector>{v});
}
