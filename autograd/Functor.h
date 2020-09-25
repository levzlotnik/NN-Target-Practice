//
// Created by LevZ on 9/12/2020.
//

#ifndef TARGETPRACTICE_FUNCTOR_H
#define TARGETPRACTICE_FUNCTOR_H

#include "VariableBase.h"
#include "blas/blas.h"

namespace autograd {

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

        /**
         * Calculates the output of the function from the inputs, and stores into output.
         * @param inputs Pointers to the inputs of the function. They are assumed to be valid for this functor.
         * @param output_ref A pointer to the output of the function
         * @return The reference to the output.
         */
        virtual void apply_forward(const vector<const Tensor<T>*> &input_ptrs,
                                   Tensor<T>* output_ptr) const noexcept = 0;

        /**
         * Calculates the gradient of the function according to the inputs and the output, and stores it into grad_ref.
         * @param input_idx The location of the input for which we calculate the gradient. It is assumed to be valid.
         * @param inputs Pointers to the inputs of the function. They are assumed to be valid for this functor.
         * @param output Pointer to the resulting output from the function.
         * @param grad_ref A pointer to the gradient tensor. It is assumed to have the shape of the corresponding input.
         * @return grad_ref
         */
        virtual void
        apply_backward(int input_idx, const vector<const Tensor<T> *> &input_ptrs, const Tensor<T> *output_ptr,
                       const Tensor<T> *output_grad_ptr, Tensor<T> *input_grad_ptr) const noexcept = 0;

        inline Tensor<T> operator()(const vector<Tensor<T>> &inputs) const {
            check_arg_shapes(get_shapes(inputs));
            Tensor<T> output(output_shape);
            vector<const Tensor<T>*> args(inputs.size());
            std::transform(inputs.begin(), inputs.end(), args.begin(), [](const Tensor<T>& t) { return &t; });
            apply_forward(args, &output);
            return output;
        }

        Variable<T> operator()(const vector<Variable<T>> &inputs, bool requires_grad = true) const;

        virtual Functor<T>* clone() const = 0;

    };

#define OVERRIDE_CLONE(functor_type) Functor<T>* clone() const override { return new functor_type(*this); }

    // Elementwise operation on a single tensor.
    template<typename T>
    class MathFunctor : public Functor<T> {
    private:
        const unary_op<T> _op;
        const unary_op<T> _dop;
        using common_math::unary_func_data<T>::get_function_data;
    public:

        inline MathFunctor(const shape_t& input_shape, const string& op_name,
                           const unary_op<T>& op, const unary_op<T>& dop) :
                Functor<T>(vector<shape_t>{input_shape}, input_shape, "ElemwiseT[" + op_name + "]"),
                _op(op), _dop(dop) {}

        /**
         * Constructor for known operations.
         * @param op_name The name of the operation.
         * @param input_shape The shape of the input tensor.
         * @note The op must be registered in common_math::unary_func_data::get_function_data, or an out_of_range \
         * will be thrown. For a non-registered op - use the 4 arguments ctor.
         */
        inline MathFunctor(const shape_t& input_shape, const string& op_name) :
            MathFunctor(input_shape, op_name,
                        get<0>(get_function_data(op_name)),
                        get<1>(get_function_data(op_name))) {}

        OVERRIDE_CLONE(MathFunctor)

        void apply_forward(const vector<const Tensor<T> *> &input_ptrs, Tensor<T> *output_ptr) const noexcept override;

        void apply_backward(int input_idx, const vector<const Tensor<T> *> &input_ptrs, const Tensor<T> *output_ptr,
                            const Tensor<T> *output_grad_ptr, Tensor<T> *input_grad_ptr) const noexcept override;
    };

    template<typename T>
    class ScalarTensorElemwiseFunctor: public Functor<T> {
    private:
        const T scalar;
        const binary_op<T> _op;
        const jac_binary_op<T> _dop;
        const bool scalar_first;
        using common_math::binary_func_data<T>::get_function_data;
    public:
        inline ScalarTensorElemwiseFunctor(const shape_t& input_shape, T scalar, const string& op_name,
                                           const binary_op<T>& op, const jac_binary_op<T>& dop, bool scalar_first) :
               Functor<T>(vector<shape_t>{input_shape}, input_shape, "ElemwiseST[" + op_name + "]"),
               scalar(scalar), _op(op), _dop(dop), scalar_first(scalar_first) {}

        inline ScalarTensorElemwiseFunctor(const shape_t& input_shape, T scalar, const string& name, bool scalar_first):
               ScalarTensorElemwiseFunctor(input_shape, scalar, name,
                   get<0>(get_function_data(name)),
                   scalar_first ? get<2>(get_function_data(name)) : get<1>(get_function_data(name)),
                   scalar_first
               ) {}
        OVERRIDE_CLONE(ScalarTensorElemwiseFunctor)

        void apply_forward(const vector<const Tensor<T> *> &input_ptrs, Tensor<T> *output_ptr) const noexcept override;

        void apply_backward(int input_idx, const vector<const Tensor<T> *> &input_ptrs, const Tensor<T> *output_ptr,
                            const Tensor<T>* output_grad_ptr, Tensor<T> *grad_ptr) const noexcept override;
    };

    template<typename T>
    class TensorTensorElemwiseFunctor : public Functor<T> {
    private:
        Tensor<T> grad_buffer;
        const binary_op<T> _op;
        const jac_binary_op<T> _dops[2];
        using common_math::binary_func_data<T>::get_function_data;

    public:
        inline TensorTensorElemwiseFunctor(const shape_t& in_shape1, const shape_t& in_shape2, const string& op_name,
                                           const binary_op<T>& op,
                                           const binary_op<T>& dop1, const binary_op<T>& dop2) :
               Functor<T>({in_shape1, in_shape2}, blas::broadcast_shapes(in_shape1, in_shape2),
                          "ElemwiseTT[" + op_name + "]"),
               _op(op), _dops{dop1, dop2}
        {
            grad_buffer = Tensor<T>(this->output_shape);
        }

        inline TensorTensorElemwiseFunctor(const shape_t& in_shape1, const shape_t& in_shape2, const string& op_name) :
        TensorTensorElemwiseFunctor(in_shape1, in_shape2, op_name,
                                    get<0>(get_function_data(op_name)),
                                    get<1>(get_function_data(op_name)), get<2>(get_function_data(op_name))) {}

        void apply_forward(const vector<const Tensor<T> *> &input_ptrs, Tensor<T> *output_ptr) const noexcept override;

        void apply_backward(int input_idx, const vector<const Tensor<T> *> &input_ptrs, const Tensor<T> *output_ptr,
                            const Tensor<T> *output_grad_ptr, Tensor<T> *input_grad_ptr) const noexcept override;

        OVERRIDE_CLONE(TensorTensorElemwiseFunctor)
    };

    template<typename T>
    class SelectFunctor : public Functor<T> {
        index_t selector_index;
        static inline shape_t get_output_shape(const shape_t& shape, const index_t& idx){
            shape_t out_shape(shape);
            if (idx.size() >= shape.size()) {
                throw runtime_error("index has more dimensions than the shape.");
            }
            for (int j=0; j < idx.size(); ++j) {
                long i = idx[j];
                size_t s = shape[j];
                normalize_index(i, s); // will throw an error if it's not ok
            }
            out_shape.erase(out_shape.begin(), out_shape.begin() + idx.size());
            return out_shape;
        }
        static inline string idx2string(const index_t& idx){
            stringstream ss;
            ss << '[';
            for (int i = 0; i < idx.size(); ++i) {
                if (i != 0)
                    ss << ", ";
                ss << idx[i];
            }
            ss << ']';
            return ss.str();
        }
    public:
        inline SelectFunctor(const shape_t& input_shape, const index_t& idx) :
               Functor<T>({input_shape}, get_output_shape(input_shape, idx),
                          "Select" + idx2string(idx)) {}
        inline SelectFunctor(const shape_t& input_shape, long idx) :
               SelectFunctor(input_shape, index_t{idx}) {}

        void apply_forward(const vector<const Tensor<T> *> &input_ptrs, Tensor<T> *output_ptr) const noexcept override;

        void apply_backward(int input_idx, const vector<const Tensor<T> *> &input_ptrs, const Tensor<T> *output_ptr,
                            const Tensor<T> *output_grad_ptr, Tensor<T> *input_grad_ptr) const noexcept override;

        OVERRIDE_CLONE(SelectFunctor)
    };

    template<typename T>
    class SliceFunctor : public Functor<T> {
        blas::SliceGroup slice_group;
    public:
        inline SliceFunctor(const shape_t& input_shape, const blas::SliceGroup& sg) :
               Functor<T>({input_shape}, sg.shape(),
                          "Slice" + sg.to_str()), slice_group(sg) {}
        inline SliceFunctor(const shape_t& input_shape, const blas::Slice& slice) :
               SliceFunctor(input_shape, blas::SliceGroup({slice}).fill_to_shape_(input_shape)) {}

        void apply_forward(const vector<const Tensor<T> *> &input_ptrs, Tensor<T> *output_ptr) const noexcept override;

        void apply_backward(int input_idx, const vector<const Tensor<T> *> &input_ptrs, const Tensor<T> *output_ptr,
                            const Tensor<T> *output_grad_ptr, Tensor<T> *input_grad_ptr) const noexcept override;

        OVERRIDE_CLONE(SliceFunctor)
    };

}



#endif //TARGETPRACTICE_FUNCTOR_H
