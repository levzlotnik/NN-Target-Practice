//
// Created by LevZ on 6/15/2020.
//

#include "Uniform.h"
#include "DistributionBase.h"
#include <cmath>

float Uniform::logp(float val) {
    return -log(upper - lower);
}

float Uniform::sample() {
    auto r = generator();
    float r_ = float(r) / float(INT32_MAX);
    return lower + r_ * (upper-lower);
}

Uniform::Uniform(float lower, float upper) : generator(random_device{}()) {
    if (lower >= upper)
        throw runtime_error("'lower' argument must be less than 'higher'.");
    this->lower = lower;
    this->upper = upper;
}

float Uniform::logcdf(float val) {
    if (val < lower || val > upper)
        return -INFINITY;
    return (val - lower) / (upper - lower);
}


ostream &Uniform::print(ostream &os, string indent, bool deep) const {
    return os << "Uniform(lower=" << lower << ", upper=" << upper << ")";
}

Uniform *Uniform::clone() const {
    return new Uniform(*this);
}

float Uniform::rsample(const vector<Vector> &inputs) const {
    check_rsample_args(inputs);
    return rsample(inputs[0][0], inputs[1][0]);
}

Uniform::sequence_type Uniform::jac_rsample(int i, const vector<Vector> &inputs, float output) const {
    check_rsample_args(inputs);
    const float dparam[] = {-1, 1};
    return { dparam[i] };
}

float Uniform::sample() const {
    auto r = default_random_engine(random_device{}())();
    float r_ = float(r) / float(INT32_MAX);
    return lower + r_ * (upper-lower);
}

float Uniform::rsample(float lower, float upper) const {
    auto r = default_random_engine(random_device{}())();
    float r_ = float(r) / float(INT32_MAX);
    return lower + r_ * (upper-lower);
}

Vector uniform(float lower, float upper, int n) {
    return Uniform(lower, upper).sample_sequence(n);
}

Matrix uniform(float lower, float upper, int n, int m) {
    return reshape(uniform(lower, upper, n*m), n, m);
}
