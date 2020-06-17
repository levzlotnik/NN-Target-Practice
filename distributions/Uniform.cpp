//
// Created by LevZ on 6/15/2020.
//

#include "Uniform.h"
#include <cmath>

float Uniform::logp(float val) {
    return -log(upper - lower);
}

float Uniform::random() {
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

