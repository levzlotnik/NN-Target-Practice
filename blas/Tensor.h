//
// Created by LevZ on 7/8/2020.
//

#ifndef BLAS_TENSOR_H_
#define BLAS_TENSOR_H_

#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "Slice.h"
#include "common_blas.h"

using namespace common_math;

// Math operations on tensors, like numpy.
namespace blas {

template <typename T>
class Tensor;

template <typename T>
class TensorView;

template <typename T>
class TensorSliced;

// template <typename T>
// class TensorSparse;

template <typename T>
class TensorTransposed;

/**
 * Fill a tensor with a single value inplace, generically.
 * @tparam Tens destination tensor type
 * @tparam T scalar type
 * @param dst destination tensor.
 * @param scalar
 * @return destination tensor.
 */
template <template <typename> class Tens, typename T>
Tens<T>& fill_(Tens<T>& dst, T scalar);

/**
 * Copy data inplace into a destination tensor, generically.
 * @tparam Tensor1 destination tensor type.
 * @tparam Tensor2 source tensor type.
 * @param dst destionation tensor.
 * @param src source tensor.
 * @return destination after copying data has been made.
 */
template <class Tensor1, class Tensor2>
Tensor1& copy_(Tensor1& dst, const Tensor2& src);

#define DECL_INTERACTIVE_ACTION_TENSOR_UNIQUE_BASE(TensorT1)                   \
    virtual TensorT1& apply_(T scalar, const binary_op<T>& op);                \
    virtual TensorT1& apply_(const unary_op<T>&);                              \
    virtual void apply(T scalar, const binary_op<T>& op, Tensor<T>& out)       \
        const;                                                                 \
    virtual void apply(T scalar, const binary_op<T>& op, TensorView<T>& out)   \
        const;                                                                 \
    virtual void apply(T scalar, const binary_op<T>& op, TensorSliced<T>& out) \
        const;                                                                 \
    virtual void apply(T scalar, const binary_op<T>& op,                       \
                       TensorTransposed<T>& out) const;                        \
    virtual Tensor<T> apply(T scalar, const binary_op<T>& op) const;           \
    virtual void apply(const unary_op<T>&, Tensor<T>& out) const;              \
    virtual void apply(const unary_op<T>&, TensorView<T>& out) const;          \
    virtual void apply(const unary_op<T>&, TensorSliced<T>& out) const;        \
    virtual void apply(const unary_op<T>&, TensorTransposed<T>& out) const;    \
    virtual Tensor<T> apply(const unary_op<T>&) const;

#define DECL_INTERACTIVE_ACTION_TENSOR_UNIQUE_OVERRIDE(TensorT1)             \
    TensorT1& apply_(T scalar, const binary_op<T>& op) override;             \
    TensorT1& apply_(const unary_op<T>&) override;                           \
    void apply(T scalar, const binary_op<T>& op, Tensor<T>& out)             \
        const override;                                                      \
    void apply(T scalar, const binary_op<T>& op, TensorView<T>& out)         \
        const override;                                                      \
    void apply(T scalar, const binary_op<T>& op, TensorSliced<T>& out)       \
        const override;                                                      \
    void apply(T scalar, const binary_op<T>& op, TensorTransposed<T>& out)   \
        const override;                                                      \
    Tensor<T> apply(T scalar, const binary_op<T>& op) const override;        \
    void apply(const unary_op<T>&, Tensor<T>& out) const override;           \
    void apply(const unary_op<T>&, TensorView<T>& out) const override;       \
    void apply(const unary_op<T>&, TensorSliced<T>& out) const override;     \
    void apply(const unary_op<T>&, TensorTransposed<T>& out) const override; \
    Tensor<T> apply(const unary_op<T>&) const override;

#define DECL_INTERACTIVE_ACTIONS_TENSOR_BASE(TensorT1, TensorT2, T)      \
    virtual TensorT1& apply_tensors_(const TensorT2<T>& other,           \
                                     const binary_op<T>& op);            \
    virtual void apply_tensors(const TensorT2<T>& other,                 \
                               const binary_op<T>& op,                   \
                               Tensor<T>& out) const;                    \
    virtual void apply_tensors(const TensorT2<T>& other,                 \
                               const binary_op<T>& op,                   \
                               TensorView<T>& out) const;                \
    virtual void apply_tensors(const TensorT2<T>& other,                 \
                               const binary_op<T>& op,                   \
                               TensorSliced<T>& out) const;              \
    virtual void apply_tensors(const TensorT2<T>& other,                 \
                               const binary_op<T>& op,                   \
                               TensorTransposed<T>& out) const;          \
    virtual Tensor<T> apply_tensors(const TensorT2<T>& other,            \
                                    const binary_op<T>& op) const;

#define DECL_INTERACTIVE_ACTIONS_TENSOR_OVERRIDE(TensorT1, TensorT2, T)        \
    TensorT1& apply_tensors_(const TensorT2<T>& other, const binary_op<T>& op) \
        override;                                                              \
    void apply_tensors(const TensorT2<T>& other, const binary_op<T>& op,       \
                       Tensor<T>& out) const override;                         \
    void apply_tensors(const TensorT2<T>& other, const binary_op<T>& op,       \
                       TensorView<T>& out) const override;                     \
    void apply_tensors(const TensorT2<T>& other, const binary_op<T>& op,       \
                       TensorSliced<T>& out) const override;                   \
    void apply_tensors(const TensorT2<T>& other, const binary_op<T>& op,       \
                       TensorTransposed<T>& out) const override;               \
    Tensor<T> apply_tensors(const TensorT2<T>& other, const binary_op<T>& op)  \
        const override;

#define MACRO_INTERACTABLE_TENSORTYPES(macro, TensorTDst, T)             \
    macro(TensorTDst, Tensor, T) macro(TensorTDst, TensorView, T) macro( \
        TensorTDst, TensorSliced, T) macro(TensorTDst, TensorTransposed, T)

#define MACRO_INTERACT_TENSORS(macro, T)                   \
    MACRO_INTERACTABLE_TENSORTYPES(macro, Tensor, T)       \
    MACRO_INTERACTABLE_TENSORTYPES(macro, TensorView, T)   \
    MACRO_INTERACTABLE_TENSORTYPES(macro, TensorSliced, T) \
    MACRO_INTERACTABLE_TENSORTYPES(macro, TensorTransposed, T)

#define DEF_ASSIGNMENT_TEMPLATE(TensorT1, TensorT2, T)     \
    inline TensorT1& copy_(const TensorT2<T>& other) {     \
        return blas::copy_(*this, other);                  \
    }                                                      \
    inline TensorT1& operator=(const TensorT2<T>& other) { \
        if (static_cast<const void*>(this) !=              \
            static_cast<const void*>(&other))              \
            this->copy_(other);                            \
        return *this;                                      \
    }

#define DEF_COPY_FILL_TEMPLATES(Tensor1, T)      \
    template <typename scalar_t>                 \
    inline Tensor1& fill_(scalar_t scalar) {     \
        return blas::fill_(*this, scalar);       \
    }                                            \
    template <typename scalar_t>                 \
    inline Tensor1& operator=(scalar_t scalar) { \
        this->fill_(scalar);                     \
        return *this;                            \
    }                                            \
    MACRO_INTERACTABLE_TENSORTYPES(DEF_ASSIGNMENT_TEMPLATE, Tensor1, T)

template <typename T>
class Tensor {
   public:
    static size_t print_precision;
    shape_t shape;
    bool is_sliced = false;

    Tensor();

    explicit Tensor(T scalar);
    explicit Tensor(const shape_t& shape);
    Tensor(std::vector<T> data, const shape_t& shape);
    Tensor(T* data, const shape_t& shape);
    Tensor(const Tensor& other);
    Tensor(Tensor&& other) noexcept;
    virtual ~Tensor();
    inline T item() const { return data[0]; }

    friend void swap(Tensor<T>& t1, Tensor<T>& t2) {
        using std::swap;
        swap(t1.size, t2.size);
        swap(t1.data, t2.data);
        swap(t1.shape, t2.shape);
        swap(t1.strides, t2.strides);
        swap(t1.requires_deletion, t2.requires_deletion);
        swap(t1.is_sliced, t2.is_sliced);
    }

    DEF_COPY_FILL_TEMPLATES(Tensor, T)

    Tensor& operator=(Tensor&& other) noexcept;

    inline Tensor& operator=(T scalar) {
        fill_(scalar);
        return *this;
    }

    template <typename Tnsr>
    class subtensor_iterator {
       private:
        T* baseline_data_ptr;
        size_t stride;
        shape_t shape;
        size_t pos;

       public:
        using difference_type = long;
        using value_type = Tnsr;
        using pointer = Tnsr*;
        using reference = Tnsr&;
        using iterator_category = std::random_access_iterator_tag;

        subtensor_iterator(T* data_ptr, size_t stride, size_t pos,
                           shape_t shape);

        inline subtensor_iterator& operator+=(difference_type n) {
            pos += n;
            return *this;
        }

        friend inline subtensor_iterator operator+(difference_type n,
                                                   subtensor_iterator i) {
            subtensor_iterator temp{i};
            return temp += n;
        }

        friend inline subtensor_iterator operator+(subtensor_iterator i,
                                                   difference_type n) {
            subtensor_iterator temp{i};
            return temp += n;
        }

        inline subtensor_iterator& operator-=(difference_type n) {
            pos -= n;
            return *this;
        }

        friend inline subtensor_iterator operator-(subtensor_iterator i,
                                                   difference_type n) {
            subtensor_iterator temp{i};
            return temp -= n;
        }

        inline difference_type operator-(subtensor_iterator other) {
            if (baseline_data_ptr != other.baseline_data_ptr)
                throw std::out_of_range(
                    "Can't operate on different baseline pointers.");
            return pos - other.pos;
        }

        inline value_type operator*() {
            Tnsr val(&baseline_data_ptr[stride * pos], shape);
            return val;
        }

        inline value_type operator[](difference_type n) { return *(*this + n); }

        inline bool operator<(subtensor_iterator other) {
            return (*this - other) < 0;
        }

        inline bool operator==(subtensor_iterator other) {
            return (*this - other) == 0;
        }

        inline bool operator!=(subtensor_iterator other) {
            return (*this - other) != 0;
        }

        inline bool operator>(subtensor_iterator other) {
            return other < *this;
        }

        inline bool operator<=(subtensor_iterator other) {
            return !(*this > other);
        }

        inline bool operator>=(subtensor_iterator other) {
            return !(*this < other);
        }

        inline subtensor_iterator& operator++() { return (*this += 1); }

        inline subtensor_iterator operator++(int) {
            subtensor_iterator temp{*this};
            ++(*this);
            return temp;
        }

        inline subtensor_iterator& operator--() { return (*this -= 1); }

        inline subtensor_iterator operator--(int) {
            subtensor_iterator temp{*this};
            --(*this);
            return temp;
        }
    };

    using iterator = subtensor_iterator<TensorView<T>>;

    // Checked indexing

    TensorView<T> operator[](long idx) const;
    TensorView<T> operator[](const index_t& index) const;
    TensorSliced<T> operator()(const Slice& slice) const;
    TensorSliced<T> operator()(const SliceGroup& slice) const;

    virtual TensorView<T> unchecked_subscript(long idx) const;  
    virtual TensorView<T> unchecked_subscript(const index_t& index) const;  
    virtual TensorSliced<T> unchecked_slice(const Slice& slice) const;  
    virtual TensorSliced<T> unchecked_slice_group(const SliceGroup& slice_group) const;
    // Returns a slice for the index.
    inline TensorSliced<T> unchecked_subscript_slice(
        const index_t& index) const {
        SliceGroup sg = SliceGroup::cover_index(index).fill_to_shape_(this->shape);
        TensorSliced ret = this->unchecked_slice_group(sg);
        // It's safe to just remove all the trailing shapes because
        // this slice is contiguous
        shape_t ret_new_shape{ret.shape.begin() + index.size(),
                              ret.shape.end()};
        ret.shape = ret_new_shape;
        return ret;
    }

    virtual Tensor contiguous() { return *this; }
    static T& get(Tensor<T>& t, size_t true_idx) { return t.data[true_idx]; }
    static T get(const Tensor<T>& t, size_t true_idx) {
        return t.data[true_idx];
    }

    // Iterate over subtensors.
    iterator begin() const;
    iterator end() const;

    // Iterate over elements.
    typedef T* eiterator;
    typedef const T* ceiterator;

    static inline eiterator elem_begin(Tensor& t) { return t.data; }
    static inline eiterator elem_end(Tensor& t) { return t.data + t.size; }

    static inline ceiterator const_elem_begin(const Tensor& t) {
        return t.data;
    }
    static inline ceiterator const_elem_end(const Tensor& t) {
        return t.data + t.size;
    }

    virtual Tensor reshape(const vector<long>& new_shape) const;
    virtual TensorView<T> view(const vector<long>& new_shape);
    virtual const TensorView<T> const_view(const vector<long>& new_shape) const;
    inline TensorView<T> view(const shape_t& new_shape) {
        return this->view(vector<long>(new_shape.begin(), new_shape.end()));
    }
    inline const TensorView<T> const_view(const shape_t& new_shape) const {
        return this->const_view(
            vector<long>(new_shape.begin(), new_shape.end()));
    }
    inline TensorView<T> unsqueeze(int dim) {
        dim = normalize_index(dim, this->dim(), true);
        shape_t new_shape(this->shape);
        new_shape.insert(new_shape.begin() + dim, 1);
        return this->view(new_shape);
    }
    inline const TensorView<T> const_unsqueeze(int dim) const {
        dim = normalize_index(dim, this->dim(), true);
        shape_t new_shape(this->shape);
        new_shape.insert(new_shape.begin() + dim, 1);
        return this->const_view(new_shape);
    }
    const TensorTransposed<T> const_permute(const shape_t& permute_idx) const;
    TensorTransposed<T> permute(const shape_t& permute_idx);

    inline size_t dim() const { return shape.size(); }

    // Declares all unary self interactions and binary interactions with a
    // constant scalar.
    DECL_INTERACTIVE_ACTION_TENSOR_UNIQUE_BASE(Tensor)

    // Declares all interactions between the different tensor types.
    MACRO_INTERACTABLE_TENSORTYPES(DECL_INTERACTIVE_ACTIONS_TENSOR_BASE, Tensor,
                                   T)

    Tensor operator-() const;

#define DEF_TENSOR_MATH_FUNC(func)                         \
    inline Tensor func() const {                           \
        return this->apply(unary_func_data<T>::func);      \
    }                                                      \
    inline void func(Tensor& out) const {                  \
        return this->apply(unary_func_data<T>::func, out); \
    }                                                      \
    inline void func(TensorView<T>& out) const {           \
        return this->apply(unary_func_data<T>::func, out); \
    }                                                      \
    inline void func(TensorSliced<T>& out) const {         \
        return this->apply(unary_func_data<T>::func, out); \
    }

#define DEF_TENSOR_MATH_FUNC_INPLACE(func) \
    inline Tensor& func##_() { return this->apply_(unary_func_data<T>::func); }

    MACRO_MATH_FUNCTIONS(DEF_TENSOR_MATH_FUNC)
    MACRO_MATH_FUNCTIONS(DEF_TENSOR_MATH_FUNC_INPLACE)

    // Reduces all elements using the same binary operation.
    Tensor<T> reduce(const binary_op<T>& op) const;
    // Reduces a dimension of elements using the same binary operation.
    Tensor<T> reduce(const binary_op<T>& op, int dim) const;
    // Reduces multiple dimensions of elements.
    Tensor<T> reduce(const binary_op<T>& op, const vector<int>& dims) const;

#define DECL_TENSOR_REDUCE_BASE(TensorOut)                                  \
    virtual void reduce(const binary_op<T>& op, TensorOut<T>& out) const;   \
    virtual void reduce(const binary_op<T>& op, int dim, TensorOut<T>& out) \
        const;                                                              \
    virtual void reduce(const binary_op<T>& op, vector<int> dims,           \
                        TensorOut<T>& out) const;

    DECL_TENSOR_REDUCE_BASE(Tensor)
    DECL_TENSOR_REDUCE_BASE(TensorView)
    DECL_TENSOR_REDUCE_BASE(TensorSliced)
    DECL_TENSOR_REDUCE_BASE(TensorTransposed)

#define DECL_TENSOR_REDUCE_OVERRIDE(TensorOut)                               \
    void reduce(const binary_op<T>& op, TensorOut<T>& out) const override;   \
    void reduce(const binary_op<T>& op, int dim, TensorOut<T>& out)          \
        const override;                                                      \
    void reduce(const binary_op<T>& op, vector<int> dims, TensorOut<T>& out) \
        const override;

#define DECL_ALL_REDUCE_OVERRIDES()           \
    DECL_TENSOR_REDUCE_OVERRIDE(Tensor)       \
    DECL_TENSOR_REDUCE_OVERRIDE(TensorView)   \
    DECL_TENSOR_REDUCE_OVERRIDE(TensorSliced) \
    DECL_TENSOR_REDUCE_OVERRIDE(TensorTransposed)

    inline Tensor sum() const {
        using b = binary_func_data<T>;
        return reduce(b::add);
    }

    inline Tensor sum(int dim) const {
        using b = binary_func_data<T>;
        return reduce(b::add, dim);
    }

    inline Tensor sum(const vector<int>& dims) {
        using b = binary_func_data<T>;
        return reduce(b::add, dims);
    }

    inline friend ostream& operator<<(ostream& os, const Tensor& t) {
        return t.print_to_os(os, true);
    }
    inline string to_str() {
        std::stringstream ss;
        ss << *this;
        return ss.str();
    }

    T* get_data_ptr();
    T* get_data_ptr() const;

    size_t size;
    shape_t strides;
    virtual ostream& print_to_os(ostream& os, bool rec_start) const;

   protected:
    static_assert(std::is_arithmetic_v<T>, "Must be an arithmetic type.");
    T* data;
    bool requires_deletion = true;

    shape_t slice2shape(const Slice& slice) const;

    Slice normalize_slice(const Slice& slice, long max_size = -1) const;

    SliceGroup normalize_slice_group(const SliceGroup& group) const;
};

template <typename T>
size_t Tensor<T>::print_precision = 0;
using DoubleTensor = Tensor<double>;
}  // namespace blas

#endif  // BLAS_TENSOR_H_