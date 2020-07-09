//
// Created by LevZ on 6/16/2020.
//

#include "Functor.h"

#include <utility>
#include "variable/AutogradVariable.h"
#include <numeric>

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
            input_shapes.size(),
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

Variable Functor::operator()(const vector<Variable>& args, bool requires_grad) const {
    check_args(args);
    auto res = Deterministic::make(name, *this, requires_grad);
    for (const auto& arg: args)
        res->add_dependency(arg);
    res->forward();
    return res;
}

Functor::Functor(const vector<int> &input_shapes, int output_shape, string name) :
        input_shapes(input_shapes), output_shape(output_shape), name(std::move(name)) {
    if (output_shape < 1 ||
        std::any_of(input_shapes.begin(), input_shapes.end(), [](int x) {return x < 1;}))
        throw runtime_error("Shapes must be positive.");
}

void Functor::check_args(const vector<Variable> &args) const {
    if (args.size() != input_shapes.size())
        throw runtime_error("Arguments count mismatch: " + to_string(args.size()) +
                            ", " + to_string(input_shapes.size()));
    for (int i=0; i<args.size(); ++i)
        if (args[i]->shape() != input_shapes[i])
            throw runtime_error("Shape Mismatch on argument " + to_string(i+1) + ": "
                                + to_string(args[i]->shape()) + ", " + to_string(input_shapes[i]));
}

Vector Slice::operator()(const vector<Vector> &args) const {
    check_args(args);
    return args[0].slice(begin, end, step);
}

Matrix Slice::jac(int i, const vector<Vector> &inputs, const Vector &output) const {
    if (i != 0)
        throw out_of_range("Only 1 element available in Slice.");
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
        throw runtime_error("The Slice size cannot be 0.");
    if (output_shape < 0)
        throw runtime_error("The Slice and step directions contradict each other: Slice direction =" +
                            to_string(e-b) + ", step = " + to_string(step));
    begin = b;
    end = e;
    this->step = step;
    const_jac = Matrix(output_shape, input_shape, true);
    for(int i=0; i<output_shape; ++i)
        const_jac(i, b + i*step) = 1;
}

Elemwise::Elemwise(string func_name, int shape) :
        Elemwise(unary_elemwise_mapping.at(func_name).first,
                 unary_elemwise_mapping.at(func_name).second, shape,
                 func_name)
{

}

Elemwise::Elemwise(unary_elemwise_t func, unary_elemwise_t dfunc, int shape, const string &func_name) :
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

Variable Elemwise::operator()(const Variable &var, bool requires_grad) const {
    return Functor::operator()({ var }, requires_grad);
}

Vector BinaryElemwise::operator()(const vector<Vector> &args) const {
    check_args(args);
    return args[0].apply(args[1], op);
}

Matrix BinaryElemwise::jac(int i, const vector<Vector> &inputs, const Vector &output) const {
    auto dxi_op = dx_op[i];
    Vector res(output_shape, 0.0f);
    for (int j=0; j < res.shape(); ++j){
        res[i] = dxi_op(inputs[0][i], inputs[1][i], output[i]);
    }
    return Matrix::diag(res, true);
}

Functor *BinaryElemwise::clone() const {
    return new BinaryElemwise(*this);
}

BinaryElemwise::BinaryElemwise(string op_name, int shape) :
    Functor({shape, shape}, shape,
            "BinaryElemwise[" + op_name + "]"),
    op(get<0>(binary_elemwise_mapping.at(op_name))),
    dx_op({get<1>(binary_elemwise_mapping.at(op_name)), get<2>(binary_elemwise_mapping.at(op_name))})
{
}

BinaryElemwise::BinaryElemwise(binary_elemwise_t op, pair<jac_binary_elemwise_t, jac_binary_elemwise_t> d_op, int shape,
                               const string &op_name) :
        Functor({shape, shape}, shape, "BinaryElemwise[" + op_name + "]"),
        op(op),
        dx_op({d_op.first, d_op.second})
{

}

Vector BinaryElemwise::operator()(const Vector &x1, const Vector &x2) const {
    return operator()({x1, x2});
}

Variable BinaryElemwise::operator()(const Variable& v1, const Variable& v2, bool requires_grad) const {
    return Functor::operator()(vector<Variable>{v1, v2}, requires_grad);
}

Vector ScalarElemwise::operator()(const vector<Vector> &args) const {
    check_args(args);
    if (scalar_first)
        return args[1].apply(args[0][0], op_for_vector);
    return args[0].apply(args[1][0], op_for_vector);
}

Matrix ScalarElemwise::jac(int i, const vector<Vector> &inputs, const Vector &output) const {
    auto dxi_op = dx_ops[i];
    bool calc_jac_of_scalar = (i == (int)(!scalar_first));
    float scalar = inputs[!scalar_first][0];
    Vector non_scalar = inputs[scalar_first];
    BinaryOperation dx_op_to_apply;
    // Create binary operation to apply on input_vector
    if (scalar_first)
        dx_op_to_apply = [scalar, dxi_op](float& input_x, float& output_x) -> float {
            return dxi_op(scalar, input_x, output_x);
        };
    else
        dx_op_to_apply = [scalar, dxi_op](float& input_x, float& output_x) -> float {
            return dxi_op(input_x, scalar, output_x);
        };

    auto res = non_scalar.apply(output, dx_op_to_apply);
    if (calc_jac_of_scalar)
        return reshape(res, -1, 1);
    return Matrix::diag(res, true);
}

Functor *ScalarElemwise::clone() const {
    return new ScalarElemwise(*this);
}

static string se_name(string op_name, bool scalar_first) {
    string args_braketed = scalar_first ? "(s, v)" : "(v, s)";
    return "ScalarElemwise[" + op_name + args_braketed + "]";
}

ScalarElemwise::ScalarElemwise(string op_name, int shape, bool scalar_first) :
        ScalarElemwise(get<0>(binary_elemwise_mapping.at(op_name)),
                       {get<1>(binary_elemwise_mapping.at(op_name)), get<2>(binary_elemwise_mapping.at(op_name))},
                       shape,
                       op_name,
                       scalar_first)
{

}

ScalarElemwise::ScalarElemwise(binary_elemwise_t op, pair<jac_binary_elemwise_t, jac_binary_elemwise_t> d_op, int shape,
                               const string &op_name, bool scalar_first) :
        Functor(scalar_first ? vector<int>{1, shape} : vector<int>{shape, 1},
            shape,
            se_name(op_name, scalar_first)),
        scalar_first(scalar_first), op(op), dx_ops({d_op.first, d_op.second})
{
    if (scalar_first) {
        op_for_vector = [op](float &x, float &y) { return op(y, x); };
        // Flip the jacobians functions.
        auto temp = dx_ops[0];
        dx_ops[0] = dx_ops[1];
        dx_ops[1] = temp;
    }
    else
        op_for_vector = op;
}

Variable ScalarElemwise::operator()(const Variable &v1, const Variable &v2, bool requires_grad) const {
    return Functor::operator()({v1, v2}, requires_grad);
}
