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
        check_args(inputs);
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

    /**
     * Applies a triop (i.e. (T, T, T) -> T) on 3 input tensors and stores in output.
     * @tparam T the dtype
     * @param out output buffer
     * @param op
     * @param in1
     * @param in2
     * @param in3
     * @note in3.shape == out.shape
     */
    template<typename T>
    static void apply_triop(Tensor<T>& out, const std::function<T (T, T, T)>& op,
                     const Tensor<T>& in1, const Tensor<T>& in2, const Tensor<T>& in3) {
        using namespace blas;
        if (in3.shape != out.shape)
            throw shape_mismatch(in3.shape, out.shape, "apply_triop");
        SliceGroup sg_in1 = SliceGroup::cover_shape(in1.shape);
        T x;
        binary_op<T> kernel = [&x, &op](T e2, T e3) -> T { return op(x, e2, e3); };
        for (const auto& idx_in1 : sg_in1){
            size_t idx_true_in1 = ravel_index(idx_in1, in1.shape, in1.size);
            x = Tensor<T>::get(in1, idx_true_in1);
            auto [sg_in2, sg_out] = broadcast_index(idx_in1, in1.shape, in2.shape, out.shape);
            TensorSliced<T> slice_in2 = const_cast<Tensor<T>&>(in2).unchecked_slice_group(sg_in2);
            TensorSliced<T> slice_in3 = const_cast<Tensor<T>&>(in3).unchecked_slice_group(sg_out);
            TensorSliced<T> slice_out = out.unchecked_slice_group(sg_out);
            slice_in2.apply_tensors(slice_in3, kernel, slice_out);
        }
    }

    inline vector<int> get_output_reduction_dims(const shape_t& input_shape, const shape_t& output_shape) {
        vector<int> ret;
        size_t dims_in = input_shape.size(), dims_out = output_shape.size();
        for (int i = 0; i < dims_in; ++i){
            int idx_in = dims_in - i - 1;
            int idx_out = dims_out - i - 1;
            if (input_shape[idx_in] == 1)
                ret.emplace_back(idx_out);
        }
        for (int i = 0; i < dims_out - dims_in; ++i)
            ret.emplace_back(i);
        return ret;
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
        apply_triop(grad_buffer, dop, in1, in2, out);
        using common_math::binary_func_data<T>::add;
        grad_buffer *= out_grad;
        vector<int> reduction_dims = get_output_reduction_dims(in_grad.shape, out_grad.shape);
        grad_buffer.reduce(add, reduction_dims, in_grad);
    }

    template<typename T>
    void
    SelectFunctor<T>::apply_forward(const vector<const Tensor<T> *> &input_ptrs, Tensor<T> *output_ptr) const noexcept {
        using blas::TensorView;
        Tensor<T>& out = *output_ptr;
        const Tensor<T>& in = *input_ptrs[0];
        TensorView<T> in_selected = in.optimized_unchecked_subscript(selector_index);
        out.copy_(in_selected);
    }

    template<typename T>
    void SelectFunctor<T>::apply_backward(int input_idx, const vector<const Tensor<T> *> &input_ptrs,
                                          const Tensor<T> *output_ptr, const Tensor<T> *output_grad_ptr,
                                          Tensor<T> *input_grad_ptr) const noexcept {
        // Just copy the gradient into the correct slice.
        using blas::TensorView;
        Tensor<T>& input_grad = *input_grad_ptr;
        const Tensor<T>& output_grad = *output_grad_ptr;
        TensorView<T> in_grad_selected = input_grad.unchecked_subscript(selector_index);
        in_grad_selected.copy_(output_grad);
    }

    template<typename T>
    void
    SliceFunctor<T>::apply_forward(const vector<const Tensor<T> *> &input_ptrs, Tensor<T> *output_ptr) const noexcept {
        using blas::TensorSliced;
        const Tensor<T>& input = *input_ptrs[0];
        Tensor<T>& output = *output_ptr;
        TensorSliced<T> input_sliced = const_cast<Tensor<T>&>(input).operator()(slice_group);
        output.copy_(input_sliced);
    }

    template<typename T>
    void SliceFunctor<T>::apply_backward(int input_idx, const vector<const Tensor<T> *> &input_ptrs,
                                         const Tensor<T> *output_ptr, const Tensor<T> *output_grad_ptr,
                                         Tensor<T> *input_grad_ptr) const noexcept {
        using blas::TensorSliced;
        Tensor<T>& input_grad = *input_grad_ptr;
        const Tensor<T>& output_grad = *output_grad_ptr;
        TensorSliced<T> in_grad_sliced = input_grad.unchecked_slice_group(slice_group);
        in_grad_sliced.copy_(output_grad);
    }
}