//
// Created by LevZ on 6/14/2020.
//

#ifndef TARGETPRACTICE_COMMON_BLAS_H
#define TARGETPRACTICE_COMMON_BLAS_H
#include <cmath>
#include <functional>
#include <string>

#include "../common.h"
#include "../common_math.h"

static long normalize_index(long i, long n, bool inclusive = false) {
    if (inclusive && i == n) return n;
    if (i < -n || i >= (n + inclusive))
        throw std::out_of_range("index should be between " +
                                std::to_string(-n) + " and " +
                                std::to_string(n - 1));
    return (i + n) % n;
}

using shape_t = std::vector<size_t>;
using index_t = std::vector<long>;

static index_t normalize_index(const index_t& index, const shape_t& shape,
                               bool inclusive = false) {
    index_t idx{index};
    int i = 0;
    for (auto& x : idx) x = normalize_index(x, shape[i++], inclusive);
    return idx;
}

std::string shape2str(const shape_t& shape);
std::string shape2str(const std::vector<long>& shape);
size_t shape2size(const shape_t& shape);

namespace blas {
using std::initializer_list;
using std::ostream;
using std::string;
using std::tuple;
using std::vector;

shape_t shape2strides(const shape_t& shape);

/**
 * Translates a dims-format index to a tuple of true index and the number of
 * elements associated with it. Checks for out-of-range errors.
 * @param idx: Unraveled index
 * @param shape: Shape of tensor
 * @param size : the total size of the tensor. Optional, if (-1) the size will
 * be determined from the shape.
 * @return pair of (true_index, num_elements).
 */
std::pair<size_t, size_t> ravel_index_checked(const index_t& idx,
                                              const shape_t& shape,
                                              int size = -1);

/**
 * @brief Translates a dims-format index to a tuple of true index and the number
 * of elements associated with it. Doesn't check for errors and only returns the
 * true index.
 * @param idx : see ravel_index_checked
 * @param shape : see ravel_index_checked
 * @param size : see ravel_index_checked
 * @return true_idx
 */
size_t ravel_index(const index_t& idx, const shape_t& shape, int size = -1);

size_t ravel_index(const index_t& idx, const shape_t& shape,
                   const shape_t& strides);

/**
 * The invers operation of ravel index - translates a true index into
 * dims-format index.
 * @param true_idx : true array index.
 * @param shape : see ravel_index_checked
 * @param size : see ravel_index_checked
 * @return unraveled index for the current tensor
 */
index_t unravel_index(size_t true_idx, const shape_t& shape, int size = -1);

class shape_mismatch : public std::out_of_range {
   public:
    shape_mismatch(const shape_t& s1, const shape_t& s2,
                   const std::string& action = "")
        : std::out_of_range(
              "Shape Mismatch: " + shape2str(s1) + ", " + shape2str(s2) +
              (action.empty() ? "" : " for action {" + action + "}") + ".") {}

    shape_mismatch(const vector<long>& s1, const shape_t& s2,
                   const std::string& action = "")
        : std::out_of_range(
              "Shape Mismatch: " + shape2str(s1) + ", " + shape2str(s2) +
              (action.empty() ? "" : " for action {" + action + "}") + ".") {}
};

class broadcast_failure : public std::out_of_range {
   public:
    using std::out_of_range::out_of_range;
    broadcast_failure(const shape_t& s1, const shape_t& s2,
                      const std::string& action = "")
        : std::out_of_range(
              "Cannot Broadcast shapes: " + shape2str(s1) + ", " +
              shape2str(s2) +
              (action.empty() ? "" : " for action {" + action + "}") + ".") {}
};
}  // namespace blas

#endif  // TARGETPRACTICE_COMMON_BLAS_H
