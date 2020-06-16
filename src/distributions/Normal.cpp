//
// Created by LevZ on 6/15/2020.
//

#include "../../include/distributions/Normal.h"
#include <cmath>

float Normal::logp(float val) {
    float delta = (val - mu)/sigma;
    return -0.5f*delta*delta - 0.5f*log(2*M_PI*sigma*sigma);
}

float Normal::random() {
    float u1 = g1.random(), u2 = g2.random();
    float z0 = sqrt(-2.0f * log(u1)) * cos(2*M_PI*u2);
    return z0 * sigma + mu;
}

Normal::Normal(float mu, float sigma) : g1(0, 1), g2(0, 1) {
    if (sigma <= 0)
        throw runtime_error("sigma must be a positive number.");
    this->mu = mu;
    this->sigma = sigma;
}


ostream &Normal::print(ostream &os, string indent, bool deep) const {
    return os << "Normal(mu=" << mu << ", sigma=" << sigma << ")";
}

Normal *Normal::clone() const {
    return new Normal(*this);
}

