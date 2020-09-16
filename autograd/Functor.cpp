//
// Created by LevZ on 9/12/2020.
//

#include "Functor.h"
#include "AutogradVariable.h"
namespace autograd {

    template<typename T>
    void Functor<T>::check_arg_shapes(const vector<shape_t> &args) const {
        using std::to_string;
        if (args.size() != input_shapes.size())
            throw std::invalid_argument(
                "Function " + name + " expects " + to_string(input_shapes.size()) +
                " arguments, got " + to_string(args.size()) + " arguments."
            );
        for (int i = 0; i < args.size(); ++i) {
            if (args[i] != input_shapes[i])
                throw std::invalid_argument(
                   "Function " + name +
                   " argument " + to_string(i) +
                   " expects input of shape " + shape2str(input_shapes[i]) +
                   ", got input of shape " + shape2str(args[i])
                );
        }
    }

    template<typename T>
    void Functor<T>::check_args(const vector<Variable<T>> &args) const {
        vector<shape_t> shapes(args.size());
        std::transform(args.begin(), args.end(), shapes.begin(), [](const Variable<T>& v) { return v.shape(); });
        check_arg_shapes(shapes);
    }

    template<typename T>
    Variable<T> Functor<T>::operator()(const vector<Variable<T>> &inputs, bool requires_grad) const {
        Variable<T> ret = AutogradVariable<T>::make(name, *this, requires_grad);
        Tensor<T>& ret_tensor = ret->data();
        apply_forward(get_tensors(inputs), &ret_tensor);
        return ret;
    }

    template<typename T>
    void
    MathFunctor<T>::apply_forward(const vector<const Tensor<T> *> &input_ptrs, Tensor<T> *output_ptr) const noexcept {
        const Tensor<T>& input = *input_ptrs[0];
        Tensor<T>& output = *output_ptr;
        input.apply(this->_op, output);
    }

    template<typename T>
    void MathFunctor<T>::apply_backward(int input_idx, const vector<const Tensor<T> *> &input_ptrs,
                                        const Tensor<T> *output_ptr,
                                        const Tensor<T> *output_grad_ptr, Tensor<T> *input_grad_ptr) const noexcept {
        // input_idx == 0 definitely.
        const Tensor<T>& input = input_ptrs[0];
        Tensor<T>& input_grad_ref = *input_grad_ptr;
        input.apply(this->_dop, input_grad_ref);
        if (output_grad_ptr)
            input_grad_ref *= *output_grad_ptr;
    }


    template<typename T>
    void ScalarTensorElemwiseFunctor<T>::apply_forward(const vector<const Tensor<T> *> &input_ptrs,
                                                       Tensor<T> *output_ptr) const noexcept {
        binary_op<T> op = _op;
        if (scalar_first)
            op = (binary_op<T>)[this](T x, T y) { return this->_op(y, x); };
        const Tensor<T>& input = *input_ptrs[0];
        Tensor<T>& output = *output_ptr;
        input.apply(scalar, op, output);
    }

    template<typename T>
    void ScalarTensorElemwiseFunctor<T>::apply_backward(int input_idx, const vector<const Tensor<T> *> &input_ptrs,
                                                        const Tensor<T> *output_ptr, const Tensor<T>* output_grad_ptr,
                                                        Tensor<T> *input_grad_ptr) const noexcept {
        if (input_idx != (!scalar_first)) // We only calculate gradients for the tensor.
            return;
        binary_op<T> dop; // takes 2 elements: (x_input, x_output) -> x_grad
        if (scalar_first)
            dop = (binary_op<T>) [this](T x, T y) { return this->_dop(this->scalar, x, y); };
        else
            dop = (binary_op<T>) [this](T x, T y) { return this->_dop(x, this->scalar, y); };
        const Tensor<T>& input = *input_ptrs[0];
        const Tensor<T>& output = *output_ptr;
        Tensor<T>& input_grad_ref = *input_grad_ptr;
        input.apply_tensors(output, dop, input_grad_ref);
        if (output_grad_ptr)
            input_grad_ref *= *output_grad_ptr;
    }


    template<typename T>
    void TensorTensorElemwiseFunctor<T>::apply_forward(const vector<const Tensor<T> *> &input_ptrs,
                                                       Tensor<T> *output_ptr) const noexcept {
        const Tensor<T>& in1 = *input_ptrs[0];
        const Tensor<T>& in2 = *input_ptrs[1];
        Tensor<T>& out = *output_ptr;
        in1.apply_tensors(in2, this->_op, out);
    }

    template<typename T>
    void TensorTensorElemwiseFunctor<T>::apply_backward(int input_idx, const vector<const Tensor<T> *> &input_ptrs,
                                                        const Tensor<T> *output_ptr, const Tensor<T> *output_grad_ptr,
                                                        Tensor<T> *input_grad_ptr) const noexcept {
        const Tensor<T>& in1 = *input_ptrs[0];
        const Tensor<T>& in2 = *input_ptrs[1];
        const Tensor<T>& out = *output_ptr;
        const Tensor<T>& out_grad = *output_grad_ptr;
        Tensor<T>& in_grad = *input_grad_ptr;
        const jac_binary_op<T>& dop = _dops[input_idx];
        // TODO - implement.
        //     Define a special function:
        //     blas::apply_tensors<Args...>(function<T(T...)> op,Tensor<T>& dst, const Args&... srcs)
        //     and use it to calculate blas::apply_tensors(dop, buffer, in1, in2, out).
        using common_math::binary_func_data<T>::add;
        grad_buffer *= out_grad;
        vector<int> reduction_dims;
        // TODO - calculate reduction_dims;
        grad_buffer.reduce(add, reduction_dims, in_grad);
    }
}