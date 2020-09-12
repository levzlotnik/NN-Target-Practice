//
// Created by LevZ on 9/12/2020.
//

#ifndef TARGETPRACTICE_FUNCTOR_H
#define TARGETPRACTICE_FUNCTOR_H

#include "VariableBase.h"
#include "blas/blas.h"

template<typename T>
inline vector<shape_t> get_shapes(const vector<Tensor<T>>& tensors) {
    vector<shape_t> ret(tensors.size());
    std::transform(tensors.begin(), tensors.end(), ret.begin(), [](const Tensor<T>& t) { return t.shape; });
    return ret;
}

template<typename T>
inline vector<const Tensor<T>*> get_tensors(const vector<Variable<T>>& variables) {
    vector<const Tensor<T>*> ret(variables.size());
    std::transform(variables.begin(), variables.end(), ret.begin(), [](const Variable<T>& v) { return &v.data(); });
    return ret;
}

namespace autograd {
    template<typename T>
    class Functor {
    public:
        const vector<shape_t> input_shapes;
        const shape_t output_shape;
        const string name;

        Functor(vector<shape_t> input_shapes, shape_t output_shape, string name) :
                input_shapes(std::move(input_shapes)), output_shape(std::move(output_shape)), name(std::move(name)) {}

        virtual ~Functor() = default;

        // Throws exception for invalid arguments.
        virtual void check_arg_shapes(const vector<shape_t> &args) const;

        void check_args(const vector<Variable<T>> &args) const;
        using InputArgType = const Tensor<T>*;
        using OutputArgType = Tensor<T>*;

        /**
         * Calculates the output of the function from the inputs, and stores into output.
         * @param inputs Pointers to the inputs of the function. They are assumed to be valid for this functor.
         * @param output_ref A pointer to the output of the function
         * @return The reference to the output.
         */
        virtual Tensor<T> &apply_forward(const vector<InputArgType> &inputs,
                                         OutputArgType output_ref) const noexcept = 0;

        /**
         * Calculates the gradient of the function according to the inputs and the output, and stores it into grad_ref.
         * @param input_idx The location of the input for which we calculate the gradient. It is assumed to be valid.
         * @param inputs Pointers to the inputs of the function. They are assumed to be valid for this functor.
         * @param output Pointer to the resulting output from the function.
         * @param grad_ref A pointer to the gradient tensor. It is assumed to have the shape of the corresponding input.
         * @return grad_ref
         */
        virtual Tensor<T> &apply_backward(int input_idx, const vector<InputArgType> &inputs, OutputArgType output,
                                          OutputArgType grad) const noexcept = 0;

        inline Tensor<T> operator()(const vector<Tensor<T>> &inputs) {
            check_arg_shapes(get_shapes(inputs));
            Tensor<T> output(output_shape);
            apply(inputs, output);
            return output;
        }

        Variable<T> operator()(const vector<Variable<T>> &inputs, bool requires_grad = true);

    };

}

// TODO - create, inherit and implement various basic Functors.

#endif //TARGETPRACTICE_FUNCTOR_H
