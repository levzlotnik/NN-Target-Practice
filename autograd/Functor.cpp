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
}