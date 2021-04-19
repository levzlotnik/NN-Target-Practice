#include "common_blas.h"
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace blas 
{

class Slice {
   private:
    class const_iterator {
        using index_t = long;
        index_t pos;
        index_t stride;

       public:
        using difference_type = long;  // unused
        using value_type = index_t;
        using pointer = const index_t*;  // unused
        using reference = const index_t&;
        using iterator_category = std::input_iterator_tag;

        const_iterator(index_t pos, index_t stride)
            : pos(pos), stride(stride) {}

        inline const_iterator& operator++() {
            pos += stride;
            return *this;
        }

        inline const const_iterator operator++(int) {
            const_iterator temp(*this);
            ++*this;
            return temp;
        }

        inline const_iterator& operator+=(difference_type x) {
            pos += x * stride;
            return *this;
        }

        inline const_iterator operator+(difference_type x) {
            const_iterator temp(*this);
            return temp += x;
        }

        inline bool operator==(const const_iterator& other) const {
            return pos == other.pos;
        }

        inline bool operator!=(const const_iterator& other) const {
            return !(*this == other);
        }

        inline reference operator*() const { return pos; }
    };

    /* why isn't abs constexpr -_- */
    static inline constexpr size_t abs(long x) { return x < 0 ? -x : x; }
    size_t size_;
    static inline constexpr size_t get_size(long b, long e, long stride) {
        return (abs(e - b) - 1) / abs(stride) + 1;
    }

   public:
    long b, e, stride;

    Slice() : Slice(0, 1, 1) {}
    Slice(long b, long e, long stride = 1);

    explicit Slice(const tuple<long, long, long>& tup)
        : Slice(std::get<0>(tup), std::get<1>(tup), std::get<2>(tup)) {}

    Slice(initializer_list<long> lst);

    constexpr void check_stride() const;

    inline size_t size() const { return size_; }

    inline const_iterator begin() const { return const_iterator(b, stride); }

    inline const_iterator end() const { return const_iterator(e, stride); }

    inline void update() { size_ = get_size(b, e, stride); }

    string to_str() const {
        using std::to_string;
        return to_string(b) + ":" + to_string(e) +
               (stride == 1 ? "" : (":" + to_string(stride)));
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
    explicit SliceGroup(const vector<tuple<int, int, int>>& slices);
    explicit SliceGroup(const vector<Slice>& slices) : slices(slices) {}
    SliceGroup(initializer_list<initializer_list<long>> lst);

    static inline SliceGroup cover_shape(const shape_t& shape) {
        SliceGroup ret;
        ret.slices.reserve(shape.size());
        for (size_t x : shape) ret.slices.emplace_back(0, x);
        return ret;
    }
    static inline SliceGroup cover_index(const index_t& index) {
        SliceGroup ret;
        ret.slices.reserve(index.size());
        for (long x : index) {
            ret.slices.emplace_back(x, x + 1);
        }
        return ret;
    }

    inline SliceGroup fill_to_shape(const shape_t& cover_shape) const {
        SliceGroup ret(*this);
        return ret.fill_to_shape_(cover_shape);
    }
    inline SliceGroup& fill_to_shape_(const shape_t& cover_shape) {
        size_t old_size = slices.size();
        slices.reserve(cover_shape.size());
        for (size_t i = old_size; i < cover_shape.size(); ++i)
            slices.emplace_back(0, cover_shape[i]);

        return *this;
    }

    class const_iterator {
        vector<Slice> slices;
        index_t pos;
        size_t elems_passed;

       public:
        using difference_type = long;  // unused
        using value_type = index_t;
        using pointer = const index_t*;  // unused
        using reference = const index_t&;
        using iterator_category = std::input_iterator_tag;

        const_iterator(index_t pos, vector<Slice> slices,
                       size_t elems_passed = 0);

        const_iterator& operator++();
        inline const_iterator operator++(int) {
            const_iterator temp(*this);
            ++*this;
            return temp;
        }

        inline bool operator==(const const_iterator& other) const {
            return elems_passed == other.elems_passed;
        }

        inline bool operator!=(const const_iterator& other) const {
            return !(*this == other);
        }

        inline reference operator*() { return pos; }
    };
    inline const_iterator begin() const {
        return const_iterator(get_init_pos(), slices, 0);
    }
    inline const_iterator end() const {
        return const_iterator(get_init_pos(), slices, size());
    }

    inline index_t translate_true_idx(size_t true_index) const {
        index_t ret(slices.size());
        for (int i = ret.size() - 1; i >= 0; --i) {
            auto s = slices[i];
            if (s.size() != 1) {
                ret[i] = s.b + (true_index % s.size()) * s.stride;
                true_index /= s.size();
            } else
                ret[i] = s.b;
        }
        return ret;
    }
    inline size_t translate_vector_idx(index_t vec_index) const {
        size_t true_index = 0;
        size_t tot_stride = 1;
        for (int i = slices.size() - 1; i >= 0; --i) {
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
        for (int i = 0; i < ret.size(); ++i) ret[i] = slices[i].size();
        return ret;
    }
    SliceGroup subslice(const SliceGroup& relative_slice) const;
    size_t size() const;
    string to_str() const;

    inline void update() {}

   private:
    inline vector<long> get_init_pos() const {
        vector<long> res(slices.size());
        for (int i = 0; i < res.size(); ++i) res[i] = slices[i].b;
        return res;
    }
};
SliceGroup broadcast_index(const index_t& src_idx, const shape_t& src_shape,
                           const shape_t& dst_shape);

/**
 * Broadcasts an index of element in tensor towards an output shape, and
 * retrieves the broadcasted second source.
 * @param src1_idx the broadcasted index.
 * @param src1_shape the shape of the first input.
 * @param src2_shape the shape of the other input.
 * @param dst_shape the shape of output.
 * @return tuple of (bc_slicegroup_src2, bc_slicegroup_dst)
 */
tuple<SliceGroup /*src2*/, SliceGroup /*dst*/> broadcast_index(
    index_t src1_idx, const shape_t& src1_shape, const shape_t& src2_shape,
    const shape_t& dst_shape);



/**
 * Broadcasts shapes to a new output shape.
 * @param s1 first input shape
 * @param s2 second input shape
 * @param safe indicator of whether or not to validate shapes.
 * @return output shape.
 */
inline shape_t broadcast_shapes(const shape_t& s1, const shape_t& s2,
                                bool safe = false) {
    if (s1.empty() || s2.empty()) return s1.empty() ? s2 : s1;
    shape_t result(std::max(s1.size(), s2.size()));
    for (int i = 0; i < result.size(); ++i) {
        size_t e1 = i >= s1.size() ? 1 : s1[s1.size() - 1 - i];
        size_t e2 = i >= s2.size() ? 1 : s2[s2.size() - 1 - i];
        if (!safe && (e1 != e2 && e1 != 1 && e2 != 1))
            throw broadcast_failure(s1, s2);
        result[result.size() - 1 - i] = std::max(e1, e2);
    }
    return result;
}

}