//
// Created by LevZ on 9/27/2020.
//

#ifndef TARGETPRACTICE_LOSS_H
#define TARGETPRACTICE_LOSS_H

#include "Functor.h"

namespace autograd {

    /**
     * Base class for loss functions.
     * Note that the output shape is always shape_t{}, i.e. Loss must always return a scalar.
     * Loss function doesn't require dependees, so the gradient w.r.t. to the output can be 1.
     * @tparam T the data type.
     */
    template<typename T>
    class Loss : public Functor<T> {
    public:
        inline Loss(const vector<shape_t>& input_shapes, const string& name) : Functor<T>(input_shapes, shape_t{}, name) {}

        virtual void forward(const vector<const Tensor<T>*>& input_ptrs, Tensor<T>& out) const = 0;
        virtual void backward(int input_idx, const vector<const Tensor<T>*>& input_ptrs, const Tensor<T>& out,
                              Tensor<T>& input_grad) const = 0;

        void apply_forward(const vector<const Tensor<T> *> &input_ptrs, Tensor<T> *output_ptr) const override {
            Tensor<T>& output = *output_ptr;
            this->forward(input_ptrs, output);
        }

        void apply_backward(int input_idx, const vector<const Tensor<T> *> &input_ptrs, const Tensor<T> *output_ptr,
                            const Tensor<T> *output_grad_ptr, Tensor<T> *input_grad_ptr) const override {
            Tensor<T> output_grad = output_grad_ptr != nullptr ?
                                    *output_grad_ptr : Tensor<T>(T(1)); // We can allow nullptr.
            Tensor<T>& input_grad = *input_grad_ptr;
            const Tensor<T>& output = *output_ptr;
            this->backward(input_idx, input_ptrs, output, input_grad);
            input_grad *= output_grad;
        }
    };

    template<typename T>
    class MSELoss : public Loss<T> {
        inline static int num_instances = 0;
        T normalization_factor;
    public:
        inline explicit MSELoss(const shape_t& input_shape) :
            Loss<T>({input_shape, input_shape},
                    "MSELoss" + to_string(num_instances++)) {
            // We normalize by the batch
            if (input_shape.empty())
                throw std::runtime_error("MSELoss can only be calculated for rank 1 (or more) tensors.");
            normalization_factor = input_shape[0];
        }

        OVERRIDE_CLONE(MSELoss)

        void forward(const vector<const Tensor<T> *> &input_ptrs, Tensor<T> &out) const override {
            const Tensor<T>& in1 = *input_ptrs[0];
            const Tensor<T>& in2 = *input_ptrs[1];
            Tensor<T>::get(out, 0) = blas::mse(in1, in2, normalization_factor);
        }

        void backward(int input_idx, const vector<const Tensor<T> *> &input_ptrs, const Tensor<T> &out,
                      Tensor<T> &input_grad) const override {
            // This loss is symmetric. We can use this fact. The gradient is of course twice the difference between
            // The tensors.
            const Tensor<T>& in1 = *input_ptrs[0];
            const Tensor<T>& in2 = *input_ptrs[1];
            T multiplier = 2.0 * (input_idx == 0 ? 1 : -1) / normalization_factor;
            using bfd = common_math::binary_func_data<T>;
            in1.apply_tensors(in2, bfd::sub, input_grad); // (input_grad = in1 - in2) [but without memory overhead]
            input_grad *= multiplier;
        }
    };

}

#endif //TARGETPRACTICE_LOSS_H
