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

    using shape_t = std::vector<size_t>;
    using index_t = std::vector<long>;

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
    std::vector<size_t> unravel_index(size_t true_idx, const shape_t &shape, int size = -1);

    std::string shape2str(const shape_t &shape);

    class shape_mismatch : public std::out_of_range {
    public:
        shape_mismatch(const shape_t &s1, const shape_t &s2, const std::string &action = "") :
                std::out_of_range(
                        "Shape Mismatch: " + shape2str(s1) + ", " + shape2str(s2) +
                        (action.empty() ? "" : " for action {" + action + "}") + "."
                ) {}
    };

    class broadcast_failure : public std::out_of_range {
    public:
        broadcast_failure(const shape_t &s1, const shape_t &s2, const std::string &action = "") :
                std::out_of_range(
                        "Cannot Broadcast shapes: " + shape2str(s1) + ", " + shape2str(s2) +
                        (action.empty() ? "" : " for action {" + action + "}") + "."
                ) {}
    };

#define DEF_COPY_FILL_TEMPLATES(Tensor1) \
    template<class Tensor2> \
    inline Tensor1& copy_(const Tensor2& other) { return copy_(*this, other); } \
    template<typename scalar_t> \
    inline Tensor1& fill_(scalar_t scalar) { return fill_(*this, scalar); } \
    template<class Tensor2> \
    inline Tensor1& operator=(const Tensor2& other) { if (this != &other) this->copy_(other); return *this; }

    /**
     * Fill a tensor with a single value inplace, generically.
     * @tparam Tens destination tensor type
     * @tparam T scalar type
     * @param dst destination tensor.
     * @param scalar
     * @return destination tensor.
     */
    template<class Tens, typename T>
    Tens &fill_(Tens &dst, T scalar);

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

    /**
     * Applies an element-wise function on all elements and stores into destination.
     * @tparam Tensor1
     * @tparam Tensor2
     * @tparam T
     * @param dst
     * @param src
     * @param op
     * @return
     */
    template<class Tensor1, class Tensor2, typename T>
    Tensor1& apply_(Tensor1& dst, const Tensor2 &src, std::function<T(T, T)> op);

    /**
     * Applies an element-wise function on all elements and stores into destination, using broadcast mechanics.
     * @tparam Tensor1
     * @tparam Tensor2
     * @tparam T
     * @param dst
     * @param src
     * @param op
     * @return
     */
    template<class Tensor1, class Tensor2, typename T>
    Tensor1& apply_broadcast_(Tensor1& dst, const Tensor2 &src, std::function<T(T, T)> op);

    /**
     * Applies an element-wise function on all elements and stores into a new tensor..
     * @tparam Tensor1
     * @tparam Tensor2
     * @tparam T
     * @param dst
     * @param src
     * @param op
     * @return
     */
    template<class Tensor1, class Tensor2, typename T>
    Tensor<T> apply(const Tensor1& src1, const Tensor2 &src2, std::function<T(T, T)> op);

    /**
     * Applies an element-wise function on all elements and stores into a new tensor, using broadcast mechanics.
     * @tparam Tensor1
     * @tparam Tensor2
     * @tparam T
     * @param dst
     * @param src
     * @param op
     * @return
     */
    template<class Tensor1, class Tensor2, typename T>
    Tensor<T> apply_broadcast(const Tensor1& src1, const Tensor2 &src2, std::function<T(T, T)> op);

#define DECL_INTERACTIVE_ACTIONS_TENSOR_BASE(TensorT1, TensorT2, T) \
    virtual TensorT1& apply_(TensorT2<T> other, binary_op<T> op); \
    virtual TensorT1& apply_broadcast_(TensorT2<T> other, binary_op<T> op); \
    virtual Tensor<T> apply(TensorT2<T> other, binary_op<T> op) const; \
    virtual Tensor<T> apply_broadcast(TensorT2<T> other, binary_op<T> op) const;

#define DECL_INTERACTIVE_ACTIONS_TENSOR_OVERRIDE(TensorT1, TensorT2, T) \
    TensorT1& apply_(TensorT2<T> other, binary_op<T> op) override; \
    TensorT1& apply_broadcast_(TensorT2<T> other, binary_op<T> op) override; \
    Tensor<T> apply(TensorT2<T> other, binary_op<T> op) const override; \
    Tensor<T> apply_broadcast(TensorT2<T> other, binary_op<T> op) const override;

#define MACRO_INTERACTABLE_TENSORTYPES(macro, TensorTDst, T) \
    macro(TensorTDst, Tensor, T) \
    macro(TensorTDst, TensorView, T) \
    macro(TensorTDst, TensorSliced, T)
    // macro(TensorTDst, TensorSparse, T) - it deserves special treatment.


    class Slice {
    private:
        class const_iterator {
            using index_t = int;
            index_t pos;
            int stride;
        public:
            using difference_type = long; // unused
            using value_type = index_t;
            using pointer = const index_t *; // unused
            using reference = const index_t &;
            using iterator_category = std::input_iterator_tag;

            const_iterator(index_t pos, int stride) : pos(pos), stride(stride) {}

            inline const_iterator &operator++() {
                pos += stride;
                return *this;
            }

            inline const_iterator operator++(int) { const_iterator temp(*this); ++*this; return temp;}

            inline const_iterator& operator+=(difference_type x) {
                pos += x*stride;
                return *this;
            }

            inline const_iterator operator+(difference_type x) { const_iterator temp(*this); return temp+=x; }

            inline bool operator==(const const_iterator &other) const { return pos == other.pos; }

            inline bool operator!=(const const_iterator &other) const { return !(*this == other); }

            inline reference operator*() { return pos; }
        };

        /* why isn't abs constexpr -_- */
        static inline constexpr size_t abs(int x) { return x < 0 ? -x: x;}

    public:
        int b, e, stride;

        Slice(int b, int e, int stride = 1);

        explicit Slice(const tuple<int, int, int> &tup) :
                b(std::get<0>(tup)), e(std::get<1>(tup)), stride(std::get<2>(tup)) {}

        constexpr Slice(initializer_list<int> lst);

        constexpr void check_stride() const;

        inline constexpr size_t size() const { return (abs(e - b) - 1) / abs(stride) + 1; }

        inline const_iterator begin() const { return const_iterator(b, stride); }

        inline const_iterator end() const { return const_iterator(e, stride); }
    };

    /**
     * An object that represents an index iterator over a Slice of a tensor.
     */
    class SliceGroup {
    public:
        explicit SliceGroup(const vector<tuple<int, int, int>> &slices);

        SliceGroup(initializer_list<initializer_list<int>> lst);

        inline size_t size() const;

        vector<Slice> slices;

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
            inline const_iterator operator ++(int) { const_iterator temp(*this); ++*this; return temp; }
            const_iterator& operator+=(difference_type x);


            inline bool operator==(const const_iterator &other) const { return elems_passed == other.elems_passed; }

            inline bool operator!=(const const_iterator &other) const { return !(*this == other); }

            inline reference operator*() { return pos; }
        };

        inline const_iterator begin() const { return const_iterator(get_init_pos(), slices, 0); }

        inline const_iterator end() const { return const_iterator(get_init_pos(), slices, size()); }

        inline index_t translate_true_idx(size_t true_index) const {
            index_t ret(slices.size());
            for (int i=0; i < ret.size(); ++i) {
                auto s = slices[i];
                if (s.size()) {
                    ret[i] = s.b + (true_index / s.size()) * s.stride;
                    true_index %= s.size();
                }
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

    private:
        inline vector<long> get_init_pos() const {
            vector<long> res;
            for (auto s: slices) res.push_back(s.b);
            return res;
        }
    };

    template<typename T>
    class Tensor {
    public:
        shape_t shape;

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
        }

        DEF_COPY_FILL_TEMPLATES(Tensor)

        virtual Tensor &operator=(Tensor &&other) noexcept;

        inline Tensor &operator=(T scalar);

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
        Tensor at(const index_t &index) const;

        template<typename ... Args>
        inline Tensor at(Args... args) const {
            return at({args...});
        }

        TensorView<T> at(const index_t &index);

        template<typename ... Args>
        inline TensorView<T> at(Args... args) {
            return at({args...});
        }

        Tensor operator[](int idx) const;
        TensorView<T> operator[](int idx);
        Tensor operator[](const index_t &index) const;
        TensorView<T> operator[](const index_t &index);
        Tensor operator()(const Slice &slice) const;
        TensorSliced<T> operator()(const Slice &slice);
        Tensor operator()(const SliceGroup &slice) const;
        TensorSliced<T> operator()(const SliceGroup &slice);

    protected:
        virtual Tensor unchecked_subscript(int idx) const; // Gets subtensor rvalue
        virtual TensorView<T> unchecked_subscript(int idx); // Gets subtensor lvalue
        virtual Tensor unchecked_subscript(const index_t &index) const; // Gets subtensor rvalue
        virtual TensorView<T> unchecked_subscript(const index_t &index); // Gets subtensor lvalue
        virtual Tensor unchecked_slice(const Slice &slice) const;   // Gets slice rvalue
        virtual TensorSliced<T> unchecked_slice(const Slice &slice); // Gets slice lvalue
        virtual Tensor unchecked_slice_group(const SliceGroup &slice_group) const;  // Gets slice rvalue
        virtual TensorSliced<T> unchecked_slice_group(const SliceGroup &slice_group); // Gets slice lvalue
        // ONLY FOR INTENRAL USE
        TensorView<T> optimized_unchecked_subscript(int idx) const;
        TensorView<T> optimized_unchecked_subscript(const index_t &index) const;

    public:

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

        Tensor reshape(shape_t new_shape);
        TensorView<T> view(shape_t new_shape);

        inline size_t dim() const { return shape.size(); }

        /*Elementwise Operations*/
        Tensor &apply_(unary_op<T> op);
        Tensor &apply_(T scalar, binary_op<T> op);

        /* Out of place operations */
        Tensor apply(unary_op<T> op) const;
        Tensor apply(T scalar, binary_op<T> op) const;

        // Declares all interactions between the different tensor types.
        MACRO_INTERACTABLE_TENSORTYPES(DECL_INTERACTIVE_ACTIONS_TENSOR_BASE, Tensor, T)
        // TODO - Define implementation in Tensor.cpp.

        Tensor operator-() const;

        // TODO - Declare all math operators in TensorMath.h

        inline friend ostream &operator<<(ostream &os, const Tensor &t) {
            return t.print_to_os(os, true);
        }
        inline string to_str() {
            std::stringstream ss;
            ss << *this;
            return ss.str();
        }

        T *get_data_ptr();

    protected:
        static_assert(std::is_arithmetic_v<T>, "Must be an arithmetic type.");
        T *data;
        shape_t strides;
        size_t size;
        bool requires_deletion = true;

        ostream &print_to_os(ostream &os, bool rec_start) const;

        shape_t slice2shape(const Slice &slice) const;

        Slice normalize_slice(const Slice &slice, int max_size = -1) const;

        SliceGroup normalize_slice_group(const SliceGroup &group) const;
    };

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

        // We don't need this here because it operates the same exact way as for Tensor.
        // // Declares all interactions between the different tensor types.
        // MACRO_INTERACTABLE_TENSORTYPES(DECL_INTERACTIVE_ACTIONS_TENSOR_OVERRIDE, TensorView, T)

        DEF_COPY_FILL_TEMPLATES(TensorView)
    };

    /**
     * A slice of the tensor - doesn't copy data from the original tensor.
     * @tparam T
     */
    template<typename T>
    class TensorSliced : public Tensor<T> {
    private:
        friend class Tensor<T>;

        const SliceGroup slice_group;

        TensorSliced(T *data, const shape_t &shape, const SliceGroup &slice_group);

        TensorSliced(Tensor<T> t, const SliceGroup &slice_group);

    public:
        ~TensorSliced() override = default;

        DEF_COPY_FILL_TEMPLATES(TensorSliced)

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
                auto idx = *sg_iterator;
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
        static inline ceiterator elem_begin(TensorSliced &ts);
        static inline ceiterator elem_end(TensorSliced &ts);
        static inline ceiterator const_elem_begin(const TensorSliced &ts);
        static inline ceiterator const_elem_end(const TensorSliced &ts);

        static T& get(TensorSliced<T>& ts, size_t true_idx);
        static T get(const TensorSliced<T>& ts, size_t true_idx);

        MACRO_INTERACTABLE_TENSORTYPES(DECL_INTERACTIVE_ACTIONS_TENSOR_OVERRIDE, TensorSliced, T)
        // TODO - Define implementation in Tensor.cpp.

    };

    template<>
    class Tensor<bool> {

    };

    template<typename T>
    class TensorSparse : public Tensor<T> {
        friend class Tensor<T>;
        // TODO - implement.
    };

    using DoubleTensor = Tensor<double>;
}


#endif //TARGETPRACTICE_TENSOR_H
