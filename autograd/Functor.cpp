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
    Variable<T> Functor<T>::operator()(const vector<Variable<T>> &inputs, bool requires_grad) {
        Variable<T> ret = AutogradVariable<T>::make(name, *this, requires_grad);
        Tensor<T>& ret_tensor = ret->data();
        apply_forward(get_tensors(inputs), &ret_tensor);
        return ret;
    }
}