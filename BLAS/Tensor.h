//
// Created by LevZ on 7/8/2020.
//

#ifndef TARGETPRACTICE_TENSOR_H
#define TARGETPRACTICE_TENSOR_H

#include <type_traits>
#include <vector>
#include <string>
#include <sstream>
#include <stdexcept>
#include "common_blas.h"

using namespace common_math;

namespace blas {
    using std::vector;
    using std::tuple;
    using std::initializer_list;
    using std::ostream;
    using std::string;

    template<typename T>
    class Tensor;

    template<typename T>
    class TensorView;

    template<typename T>
    class TensorSliced;

    template<typename T>
    class TensorSparse;

    shape_t shape2strides(const shape_t &shape);

    /**
     *
     * Translates a dims-format index to a tuple of true index and the number of elements associated with it.
     * Checks for out-of-range errors.
     * @param idx: Unraveled index
     * @param shape: Shape of tensor
     * @param size : the total size of the tensor. Optional, if (-1) the size will be determined from the shape.
     * @return pair of (true_index, num_elements).
     */
    std::pair<size_t, size_t> ravel_index_checked(const index_t &idx, const shape_t &shape, int size = -1);

    /**
     * Same as last function but doesn't check for errors and only returns the true index.
     * @param idx : see ravel_index_checked
     * @param shape : see ravel_index_checked
     * @param size : see ravel_index_checked
     * @return
     */
    size_t ravel_index(const index_t &idx, const shape_t &shape, int size = -1);

    /**
     * The invers operation of ravel index - translates a true index into dims-format index.
     * @param true_idx : true array index.
     * @param shape : see ravel_index_checked
     * @param size : see ravel_index_checked
     * @return unraveled index for the current tensor
     */
    index_t unravel_index(size_t true_idx, const shape_t &shape, int size = -1);

    std::string shape2str(const shape_t &shape);
    std::string shape2str(const vector<long>& shape);
    size_t shape2size(const shape_t& shape);

    class shape_mismatch : public std::out_of_range {
    public:
        shape_mismatch(const shape_t &s1, const shape_t &s2, const std::string &action = "") :
                std::out_of_range(
                        "Shape Mismatch: " + shape2str(s1) + ", " + shape2str(s2) +
                        (action.empty() ? "" : " for action {" + action + "}") + "."
                ) {}

        shape_mismatch(const vector<long> &s1, const shape_t &s2, const std::string &action = "") :
                std::out_of_range(
                        "Shape Mismatch: " + shape2str(s1) + ", " + shape2str(s2) +
                        (action.empty() ? "" : " for action {" + action + "}") + "."
                ) {}
    };

    class broadcast_failure : public std::out_of_range {
    public:
        using std::out_of_range::out_of_range;
        broadcast_failure(const shape_t &s1, const shape_t &s2, const std::string &action = "") :
                std::out_of_range(
                        "Cannot Broadcast shapes: " + shape2str(s1) + ", " + shape2str(s2) +
                        (action.empty() ? "" : " for action {" + action + "}") + "."
                ) {}
    };

    /**
     * Fill a tensor with a single value inplace, generically.
     * @tparam Tens destination tensor type
     * @tparam T scalar type
     * @param dst destination tensor.
     * @param scalar
     * @return destination tensor.
     */
    template<template<typename>class Tens, typename T>
    Tens<T> &fill_(Tens<T> &dst, T scalar);

    /**
     * Copy data inplace into a destination tensor, generically.
     * @tparam Tensor1 destination tensor type.
     * @tparam Tensor2 source tensor type.
     * @param dst destionation tensor.
     * @param src source tensor.
     * @return destination after copying data has been made.
     */
    template<class Tensor1, class Tensor2>
    Tensor1 &copy_(Tensor1 &dst, const Tensor2 &src);


#define DECL_INTERACTIVE_ACTION_TENSOR_UNIQUE_BASE(TensorT1) \
    virtual TensorT1& apply_(T scalar, const binary_op<T>& op); \
    virtual TensorT1& apply_(const unary_op<T>&); \
    virtual void apply(T scalar, const binary_op<T>& op, Tensor<T>& out) const; \
    virtual void apply(T scalar, const binary_op<T>& op, TensorView<T>& out) const; \
    virtual void apply(T scalar, const binary_op<T>& op, TensorSliced<T>& out) const; \
    virtual Tensor<T> apply(T scalar, const binary_op<T>& op) const; \
    virtual void apply(const unary_op<T>&, Tensor<T>& out) const; \
    virtual void apply(const unary_op<T>&, TensorView<T>& out) const; \
    virtual void apply(const unary_op<T>&, TensorSliced<T>& out) const; \
    virtual Tensor<T> apply(const unary_op<T>&) const;

#define DECL_INTERACTIVE_ACTION_TENSOR_UNIQUE_OVERRIDE(TensorT1) \
    TensorT1& apply_(T scalar, const binary_op<T>& op) override; \
    TensorT1& apply_(const unary_op<T>&) override; \
    void apply(T scalar, const binary_op<T>& op, Tensor<T>& out) const override; \
    void apply(T scalar, const binary_op<T>& op, TensorView<T>& out) const override; \
    void apply(T scalar, const binary_op<T>& op, TensorSliced<T>& out) const override; \
    Tensor<T> apply(T scalar, const binary_op<T>& op) const override; \
    void apply(const unary_op<T>&, Tensor<T>& out) const override; \
    void apply(const unary_op<T>&, TensorView<T>& out) const override; \
    void apply(const unary_op<T>&, TensorSliced<T>& out) const override; \
    Tensor<T> apply(const unary_op<T>&) const override;

#define DECL_INTERACTIVE_ACTIONS_TENSOR_BASE(TensorT1, TensorT2, T) \
    virtual TensorT1& apply_tensors_(TensorT2<T> other, const binary_op<T>& op); \
    virtual void apply_tensors(TensorT2<T> other, const binary_op<T>& op, Tensor<T>& out) const; \
    virtual void apply_tensors(TensorT2<T> other, const binary_op<T>& op, TensorView<T>& out) const; \
    virtual void apply_tensors(TensorT2<T> other, const binary_op<T>& op, TensorSliced<T>& out) const; \
    virtual Tensor<T> apply_tensors(TensorT2<T> other, const binary_op<T>& op) const;

#define DECL_INTERACTIVE_ACTIONS_TENSOR_OVERRIDE(TensorT1, TensorT2, T) \
    TensorT1& apply_tensors_(TensorT2<T> other, const binary_op<T>& op)override; \
    void apply_tensors(TensorT2<T> other, const binary_op<T>& op, Tensor<T>& out) const override; \
    void apply_tensors(TensorT2<T> other, const binary_op<T>& op, TensorView<T>& out) const override; \
    void apply_tensors(TensorT2<T> other, const binary_op<T>& op, TensorSliced<T>& out) const override; \
    Tensor<T> apply_tensors(TensorT2<T> other, const binary_op<T>& op) const override;

#define MACRO_INTERACTABLE_TENSORTYPES(macro, TensorTDst, T) \
    macro(TensorTDst, Tensor, T) \
    macro(TensorTDst, TensorView, T) \
    macro(TensorTDst, TensorSliced, T)
    // macro(TensorTDst, TensorSparse, T) - it deserves special treatment.

#define MACRO_INTERACT_TENSORS(macro, T) \
    MACRO_INTERACTABLE_TENSORTYPES(macro, Tensor, T)\
    MACRO_INTERACTABLE_TENSORTYPES(macro, TensorView, T)\
    MACRO_INTERACTABLE_TENSORTYPES(macro, TensorSlice, T)\

#define DEF_ASSIGNMENT_TEMPLATE(TensorT1, TensorT2, T) \
    inline TensorT1& copy_(const TensorT2<T>& other) { return blas::copy_(*this, other); } \
    inline TensorT1& operator=(const TensorT2<T>& other) { \
        if (static_cast<const void*>(this) != static_cast<const void*>(&other)) this->copy_(other); \
        return *this; \
    }

#define DEF_COPY_FILL_TEMPLATES(Tensor1, T) \
    template<typename scalar_t> \
    inline Tensor1& fill_(scalar_t scalar) { return blas::fill_(*this, scalar); } \
    template<typename scalar_t> \
    inline Tensor1& operator=(scalar_t scalar) { this->fill_(scalar); return *this;} \
    MACRO_INTERACTABLE_TENSORTYPES(DEF_ASSIGNMENT_TEMPLATE, Tensor1, T)


    class Slice {
    private:
        class const_iterator {
            using index_t = long;
            index_t pos;
            index_t stride;
        public:
            using difference_type = long; // unused
            using value_type = index_t;
            using pointer = const index_t *; // unused
            using reference = const index_t &;
            using iterator_category = std::input_iterator_tag;

            const_iterator(index_t pos, index_t stride) : pos(pos), stride(stride) {}

            inline const_iterator &operator++() {
                pos += stride;
                return *this;
            }

            inline const const_iterator operator++(int) { const_iterator temp(*this); ++*this; return temp;}

            inline const_iterator& operator+=(difference_type x) {
                pos += x*stride;
                return *this;
            }

            inline const_iterator operator+(difference_type x) { const_iterator temp(*this); return temp+=x; }

            inline bool operator==(const const_iterator &other) const { return pos == other.pos; }

            inline bool operator!=(const const_iterator &other) const { return !(*this == other); }

            inline reference operator*() const { return pos; }
        };

        /* why isn't abs constexpr -_- */
        static inline constexpr size_t abs(long x) { return x < 0 ? -x: x;}
        size_t size_;
        static inline constexpr size_t get_size(long b, long e, long stride) { return (abs(e-b)-1) / abs(stride) + 1; }
    public:
        long b, e, stride;

        Slice() : Slice(0, 0, 1) {}
        Slice(long b, long e, long stride = 1);

        explicit Slice(const tuple<long, long, long> &tup) :
                Slice(std::get<0>(tup),std::get<1>(tup), std::get<2>(tup)) {}

        Slice(initializer_list<long> lst);

        constexpr void check_stride() const;

        inline size_t size() const { return size_; }

        inline const_iterator begin() const { return const_iterator(b, stride); }

        inline const_iterator end() const { return const_iterator(e, stride); }

        inline void update() { size_ = get_size(b, e, stride); }

        string to_str() const {
            using std::to_string;
            return to_string(b) + ":" + to_string(e) + (stride == 1 ? "" : (":" + to_string(stride)));
        }

        Slice subslice(const Slice& relative_slice) const;
    };

    /**
     * An object that represents an index iterator over a Slice of a tensor.
     */
    class SliceGroup {
    public:
        vector<Slice> slices;
        SliceGroup() = default;
        SliceGroup(size_t dims) : slices(dims) {}
        explicit SliceGroup(const vector<tuple<int, int, int>> &slices);
        explicit SliceGroup(const vector<Slice>& slices) : slices(slices) {}
        SliceGroup(initializer_list<initializer_list<long>> lst);

        static inline SliceGroup cover_shape(const shape_t& shape) {
            SliceGroup ret;
            ret.slices.resize(shape.size());
            int i=0;
            for (auto& slice: ret.slices) {
                slice.e = shape[i++];
                slice.update();
            }
            return ret;
        }
        static inline SliceGroup cover_index(const index_t& index) {
            SliceGroup ret;
            ret.slices.resize(index.size());
            int i = 0;
            for (auto& slice : ret.slices) {
                long x = index[i++];
                slice.b = x;
                slice.e = x + 1;
                slice.update();
            }
            return ret;
        }

        inline SliceGroup fill_to_shape(const shape_t& cover_shape) const {
            SliceGroup ret(*this);
            return ret.fill_to_shape_(cover_shape);
        }
        inline SliceGroup& fill_to_shape_(const shape_t& cover_shape) {
            size_t old_size = slices.size();
            slices.resize(cover_shape.size());
            for (size_t i = old_size; i < cover_shape.size(); ++i) {
                slices[i].e = cover_shape[i];
                slices[i].update();
            }
            return *this;
        }

        class const_iterator {
            vector<Slice> slices;
            index_t pos;
            size_t elems_passed;
        public:
            using difference_type = long; // unused
            using value_type = index_t;
            using pointer = const index_t *; // unused
            using reference = const index_t &;
            using iterator_category = std::input_iterator_tag;

            const_iterator(index_t pos, vector<Slice> slices, size_t elems_passed = 0);

            const_iterator &operator++();
            inline const_iterator operator ++(int) { const_iterator temp(*this); ++*this; return temp; }\

            inline bool operator==(const const_iterator &other) const { return elems_passed == other.elems_passed; }

            inline bool operator!=(const const_iterator &other) const { return !(*this == other); }

            inline reference operator*() { return pos; }
        };
        inline const_iterator begin() const { return const_iterator(get_init_pos(), slices, 0); }
        inline const_iterator end() const { return const_iterator(get_init_pos(), slices, size()); }

        inline index_t translate_true_idx(size_t true_index) const {
            index_t ret(slices.size());
            for (int i=ret.size()-1; i >= 0; --i) {
                auto s = slices[i];
                if (s.size() != 1) {
                    ret[i] = s.b + (true_index % s.size()) * s.stride;
                    true_index /= s.size();
                }
                else ret[i] = s.b;
            }
            return ret;
        }
        inline size_t translate_vector_idx(index_t vec_index) const {
            size_t true_index = 0;
            size_t tot_stride = 1;
            for (int i = slices.size() - 1 ; i >= 0; --i ){
                auto s = slices[i];
                if (s.size()) {
                    long p = vec_index[i];
                    size_t pos = (p - s.b) / s.stride;
                    true_index += pos * tot_stride;
                    tot_stride *= s.size();
                }
            }
            return true_index;
        }
        inline shape_t shape() const {
            shape_t ret(slices.size());
            for (int i=0; i < ret.size(); ++i)
                ret[i] = slices[i].size();
            return ret;
        }
        SliceGroup subslice(const SliceGroup& relative_slice) const;
        size_t size() const;
        string to_str() const;

        inline void update() { }

    private:
        inline vector<long> get_init_pos() const {
            vector<long> res(slices.size());
            for (int i=0; i < res.size(); ++i)
                res[i] = slices[i].b;
            return res;
        }
    };

    template<typename T>
    class Tensor {
    public:
        static size_t print_precision;
        shape_t shape;
        bool is_sliced = false;

        Tensor();

        explicit Tensor(T scalar);
        explicit Tensor(const shape_t &shape);
        Tensor(std::vector<T> data, const shape_t &shape);
        Tensor(T *data, const shape_t &shape);
        Tensor(const Tensor &other);
        Tensor(Tensor &&other) noexcept;
        virtual ~Tensor();
        template<typename T_>
        friend void swap(Tensor<T_> &t1, Tensor<T_> &t2) {
            using std::swap;
            swap(t1.size, t2.size);
            swap(t1.data, t2.data);
            swap(t1.shape, t2.shape);
            swap(t1.strides, t2.strides);
            swap(t1.requires_deletion, t2.requires_deletion);
            swap(t1.is_sliced, t2.is_sliced);
        }

        DEF_COPY_FILL_TEMPLATES(Tensor, T)

        Tensor &operator=(Tensor &&other) noexcept;

        inline Tensor &operator=(T scalar) { fill_(scalar); return *this;}

        template<typename Tnsr>
        class subtensor_iterator {
        private:
            T *baseline_data_ptr;
            size_t stride;
            shape_t shape;
            size_t pos;
        public:
            using difference_type = long;
            using value_type = Tnsr;
            using pointer = Tnsr *;
            using reference = Tnsr &;
            using iterator_category = std::random_access_iterator_tag;

            subtensor_iterator(T *data_ptr, size_t stride, size_t pos, shape_t shape);

            inline subtensor_iterator &operator+=(difference_type n) {
                pos += n;
                return *this;
            }

            friend inline subtensor_iterator operator+(difference_type n, subtensor_iterator i) {
                subtensor_iterator temp{i};
                return temp += n;
            }

            friend inline subtensor_iterator operator+(subtensor_iterator i, difference_type n) {
                subtensor_iterator temp{i};
                return temp += n;
            }

            inline subtensor_iterator &operator-=(difference_type n) {
                pos -= n;
                return *this;
            }

            friend inline subtensor_iterator operator-(subtensor_iterator i, difference_type n) {
                subtensor_iterator temp{i};
                return temp -= n;
            }

            inline difference_type operator-(subtensor_iterator other) {
                if (baseline_data_ptr != other.baseline_data_ptr)
                    throw std::out_of_range("Can't operate on different baseline pointers.");
                return pos - other.pos;
            }

            inline value_type operator*() {
                Tnsr val(&baseline_data_ptr[stride * pos], shape);
                return val;
            }

            inline value_type operator[](difference_type n) { return *(*this + n); }

            inline bool operator<(subtensor_iterator other) { return (*this - other) < 0; }

            inline bool operator==(subtensor_iterator other) { return (*this - other) == 0; }

            inline bool operator!=(subtensor_iterator other) { return (*this - other) != 0; }

            inline bool operator>(subtensor_iterator other) { return other < *this; }

            inline bool operator<=(subtensor_iterator other) { return !(*this > other); }

            inline bool operator>=(subtensor_iterator other) { return !(*this < other); }

            inline subtensor_iterator &operator++() { return (*this += 1); }

            inline subtensor_iterator operator++(int) {
                subtensor_iterator temp{*this};
                ++(*this);
                return temp;
            }

            inline subtensor_iterator &operator--() { return (*this -= 1); }

            inline subtensor_iterator operator--(int) {
                subtensor_iterator temp{*this};
                --(*this);
                return temp;
            }
        };

        using iterator = subtensor_iterator<TensorView<T>>;
        using const_iterator = subtensor_iterator<Tensor<T>>;

        // Checked indexing

        Tensor operator[](long idx) const;
        TensorView<T> operator[](long idx);
        Tensor operator[](const index_t &index) const;
        TensorView<T> operator[](const index_t &index);
        Tensor operator()(const Slice &slice) const;
        TensorSliced<T> operator()(const Slice &slice);
        Tensor operator()(const SliceGroup &slice) const;
        TensorSliced<T> operator()(const SliceGroup &slice);

        virtual Tensor unchecked_subscript(long idx) const; // Gets subtensor rvalue
        virtual TensorView<T> unchecked_subscript(long idx); // Gets subtensor lvalue
        virtual Tensor unchecked_subscript(const index_t &index) const; // Gets subtensor rvalue
        virtual TensorView<T> unchecked_subscript(const index_t &index); // Gets subtensor lvalue
        virtual Tensor unchecked_slice(const Slice &slice) const;   // Gets slice rvalue
        virtual TensorSliced<T> unchecked_slice(const Slice &slice); // Gets slice lvalue
        virtual Tensor unchecked_slice_group(const SliceGroup &slice_group) const;  // Gets slice rvalue
        virtual TensorSliced<T> unchecked_slice_group(const SliceGroup &slice_group); // Gets slice lvalue

        // Returns a slice for the index.
        inline TensorSliced<T> unchecked_subscript_slice(const index_t& index) {
            SliceGroup sg = SliceGroup::cover_index(index).fill_to_shape_(this->shape);
            TensorSliced ret = unchecked_slice_group(sg);
            // It's safe to just remove all the trailing shapes because
            // this slice is contiguous
            shape_t ret_new_shape{ret.shape.begin() + index.size(), ret.shape.end()};
            ret.shape = ret_new_shape;
            return ret;
        }
        // Returns a slice for the index.
        inline TensorSliced<T> unchecked_subscript_slice(const index_t& index) const {
            SliceGroup sg = SliceGroup::cover_index(index).fill_to_shape_(this->shape);
            TensorSliced ret = const_cast<Tensor&>(*this).unchecked_slice_group(sg);
            // It's safe to just remove all the trailing shapes because
            // this slice is contiguous
            shape_t ret_new_shape{ret.shape.begin() + index.size(), ret.shape.end()};
            ret.shape = ret_new_shape;
            return ret;
        }
        // ONLY FOR INTENRAL USE
        TensorView<T> optimized_unchecked_subscript(int idx) const;
        TensorView<T> optimized_unchecked_subscript(const index_t &index) const;

        virtual Tensor contiguous() { return *this; }
        static T& get(Tensor<T>& t, size_t true_idx) { return t.data[true_idx]; }
        static T get(const Tensor<T>& t, size_t true_idx) { return t.data[true_idx]; }

        // Iterate over subtensors.
        iterator begin();
        iterator end();

        const_iterator begin() const;
        const_iterator end() const;

        // Iterate over elements.
        typedef T *eiterator;
        typedef const T *ceiterator;

        static inline eiterator elem_begin(Tensor &t) { return t.data; }
        static inline eiterator elem_end(Tensor &t) { return t.data + t.size; }

        static inline ceiterator const_elem_begin(const Tensor &t) { return t.data; }
        static inline ceiterator const_elem_end(const Tensor &t) { return t.data + t.size; }

        virtual Tensor reshape(const vector<long> &new_shape) const;
        virtual TensorView<T> view(const vector<long> &new_shape);
        virtual TensorView<T> const_view(const vector<long>& new_shape) const;
        inline TensorView<T> view(const shape_t& new_shape) {
            return this->view(vector<long>(new_shape.begin(), new_shape.end()));
        }
        inline TensorView<T> const_view(const shape_t& new_shape) const {
            return this->const_view(vector<long>(new_shape.begin(), new_shape.end()));
        }
        inline TensorView<T> unsqueeze(int dim) {
            dim = normalize_index(dim, this->dim(), true);
            shape_t new_shape(this->shape);
            new_shape.insert(new_shape.begin() + dim, 1);
            return this->view(new_shape);
        }
        inline TensorView<T> const_unsqueeze(int dim) const{
            dim = normalize_index(dim, this->dim(), true);
            shape_t new_shape(this->shape);
            new_shape.insert(new_shape.begin() + dim, 1);
            return this->const_view(new_shape);
        }

        inline size_t dim() const { return shape.size(); }

        // Declares all unary self interactions and binary interactions with a constant scalar.
        DECL_INTERACTIVE_ACTION_TENSOR_UNIQUE_BASE(Tensor)

        // Declares all interactions between the different tensor types.
        MACRO_INTERACTABLE_TENSORTYPES(DECL_INTERACTIVE_ACTIONS_TENSOR_BASE, Tensor, T)

        Tensor operator-() const;

#define DEF_TENSOR_MATH_FUNC(func) \
        inline Tensor func() const { return this->apply(unary_func_data<T>::func); }

#define DEF_TENSOR_MATH_FUNC_INPLACE(func) \
        inline Tensor& func ##_() { return this->apply_(unary_func_data<T>::func); }

        MACRO_MATH_FUNCTIONS(DEF_TENSOR_MATH_FUNC)
        MACRO_MATH_FUNCTIONS(DEF_TENSOR_MATH_FUNC_INPLACE)


        inline friend ostream &operator<<(ostream &os, const Tensor &t) {
            return t.print_to_os(os, true);
        }
        inline string to_str() {
            std::stringstream ss;
            ss << *this;
            return ss.str();
        }

        T *get_data_ptr();
        T *get_data_ptr() const;

        size_t size;
        virtual ostream &print_to_os(ostream &os, bool rec_start) const;


    protected:
        static_assert(std::is_arithmetic_v<T>, "Must be an arithmetic type.");
        T *data;
        shape_t strides;
        bool requires_deletion = true;

        shape_t slice2shape(const Slice &slice) const;

        Slice normalize_slice(const Slice &slice, long max_size = -1) const;

        SliceGroup normalize_slice_group(const SliceGroup &group) const;

    };

    template <typename T>
    size_t Tensor<T>::print_precision = 0;
    /**
     * A view of a tensor - doesn't copy data from original tensor.
     * @tparam T : stored data type.
     */
    template<typename T>
    class TensorView : public Tensor<T> {
        friend class Tensor<T>;

        TensorView(T *data, const std::vector<size_t> &);

        explicit TensorView(Tensor<T> t);

    public:
        ~TensorView() override = default; // Doesn't delete the data.
        using eiterator = T *;
        using ceiterator = const T *;

        static T& get(TensorView<T>& tv, size_t true_idx) { return tv.data[true_idx]; }
        static T get(const TensorView<T>& tv, size_t true_idx) { return tv.data[true_idx]; }

        inline Tensor<T> contiguous() override {
            Tensor<T> t(this->shape);
            t.copy_(*this);
            return t;
        }

        // We don't need this here because it operates the same exact way as for Tensor.
        // // Declares all interactions between the different tensor types.
        // MACRO_INTERACTABLE_TENSORTYPES(DECL_INTERACTIVE_ACTIONS_TENSOR_OVERRIDE, TensorView, T)

        DECL_INTERACTIVE_ACTION_TENSOR_UNIQUE_OVERRIDE(TensorView)
        DEF_COPY_FILL_TEMPLATES(TensorView, T)
    };

    /**
     * A slice of the tensor - doesn't copy data from the original tensor.
     * @tparam T
     */
    template<typename T>
    class TensorSliced : public Tensor<T> {
    private:
        friend class Tensor<T>;
        shape_t underlying_tensor_shape;
        const size_t underlying_tensor_size;
        SliceGroup slice_group;

        TensorSliced(T *data, const shape_t &shape, const SliceGroup &slice_group);

        TensorSliced(Tensor<T>& t, const SliceGroup &slice_group);
        static TensorSliced _from_const(const Tensor<T>& t, const SliceGroup &slice_group);

    public:
        ~TensorSliced() override = default;
        TensorSliced(const TensorSliced& other);
        TensorSliced(TensorSliced&& other)  noexcept = default;

        DECL_INTERACTIVE_ACTION_TENSOR_UNIQUE_OVERRIDE(TensorSliced)
        DEF_COPY_FILL_TEMPLATES(TensorSliced, T)

        template<typename T_>
        struct eliterator {
            T_ *data;
            SliceGroup::const_iterator sg_iterator;
            const shape_t shape;
            const size_t size;

            using difference_type = ptrdiff_t;
            using value_type = T_;
            using reference = T_ &;
            using pointer = T_ *;
            using iterator_category = std::forward_iterator_tag;

            inline reference operator*() {
                const auto& idx = *sg_iterator;
                auto true_idx = ravel_index(idx, shape, size);
                return data[true_idx];
            }

            inline eliterator &operator++() {
                ++sg_iterator;
                return *this;
            }

            inline eliterator operator++(int) {
                eliterator temp(*this);
                ++(*this);
                return temp;
            }

            inline bool operator==(const eliterator &other) { return sg_iterator == other.sg_iterator; }

            inline bool operator!=(const eliterator &other) { return !(*this == other); }
        };
        using eiterator = eliterator<T>;
        using ceiterator = eliterator<const T>;
        static inline eiterator elem_begin(TensorSliced &ts);
        static inline eiterator elem_end(TensorSliced &ts);
        static inline ceiterator const_elem_begin(const TensorSliced &ts);
        static inline ceiterator const_elem_end(const TensorSliced &ts);

        Tensor<T> unchecked_subscript(long idx) const override;
        Tensor<T> unchecked_subscript(const index_t &index) const override;
        /* We mark these as forbidden because we cannot create TensorView out of TensorSlice.*/

        MARK_FORBIDDEN(TensorView<T> unchecked_subscript(long idx) override)
        MARK_FORBIDDEN(TensorView<T> unchecked_subscript(const index_t &index) override)
        Tensor<T> unchecked_slice(const Slice &slice) const override;
        TensorSliced<T> unchecked_slice(const Slice &slice) override;
        Tensor<T> unchecked_slice_group(const SliceGroup &slice_group) const override;
        TensorSliced<T> unchecked_slice_group(const SliceGroup &slice_group) override;
        MARK_FORBIDDEN(TensorView<T> view(const vector<long>& new_shape) override)

        Tensor<T> reshape(const vector<long> &new_shape) const override;

        Tensor<T> contiguous() override;

    protected:
        ostream &print_to_os(ostream &os, bool rec_start) const override;

    public:

        static T& get(TensorSliced<T>& ts, size_t true_idx);
        static T get(const TensorSliced<T>& ts, size_t true_idx);

        MACRO_INTERACTABLE_TENSORTYPES(DECL_INTERACTIVE_ACTIONS_TENSOR_OVERRIDE, TensorSliced, T)

        Tensor<T> contiguous() const;

        TensorSliced slice_unsqueeze(int i);
        TensorSliced<T> const_slice_unsqueeze(int i) const;
    };

    SliceGroup broadcast_index(const index_t &src_idx, const shape_t &src_shape, const shape_t &dst_shape);

    /**
     * Broadcasts an index of element in tensor towards an output shape, and retrieves the broadcasted
     * second source.
     * @param src1_idx the broadcasted index.
     * @param src1_shape the shape of the first input.
     * @param src2_shape the shape of the other input.
     * @param dst_shape the shape of output.
     * @return tuple of (bc_slicegroup_src2, bc_slicegroup_dst)
     */
    tuple<SliceGroup /*src2*/, SliceGroup /*dst*/>
    broadcast_index(index_t src1_idx, const shape_t& src1_shape, const shape_t& src2_shape, const shape_t& dst_shape);

    template<>
    class Tensor<bool> {

    };

    template<typename T>
    class TensorSparse : public Tensor<T> {
        friend class Tensor<T>;
        // TODO - implement.
    };

    using DoubleTensor = Tensor<double>;

    /**
     * Broadcasts shapes to a new output shape.
     * @param s1 first input shape
     * @param s2 second input shape
     * @param safe indicator of whether or not to validate shapes.
     * @return output shape.
     */
    inline shape_t broadcast_shapes(const shape_t &s1, const shape_t &s2, bool safe=false) {
        if (s1.empty() || s2.empty())
            return s1.empty() ? s2 : s1;
        shape_t result(std::max(s1.size(), s2.size()));
        for (int i=0; i < result.size(); ++i) {
            size_t e1 = i >= s1.size() ? 1 : s1[s1.size() - 1 - i];
            size_t e2 = i >= s2.size() ? 1 : s2[s2.size() - 1 - i];
            if (!safe && (e1 != e2 && e1 != 1 && e2 != 1))
                throw broadcast_failure(s1, s2);
            result[result.size()-1-i] = std::max(e1, e2);
        }
        return result;
    }
}


#endif //TARGETPRACTICE_TENSOR_H
