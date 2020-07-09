//
// Created by LevZ on 6/15/2020.
//

#include "Normal.h"
#include "DistributionBase.h"
#define _USE_MATH_DEFINES
#include <math.h>

float Normal::logp(float val) {
    float delta = (val - mu)/sigma;
    return -0.5f*delta*delta - 0.5f*log(2*M_PI*sigma*sigma);
}

float Normal::sample_base() const {
    float u1 = g1.sample(), u2 = g2.sample();
    float z0 = sqrt(-2.0f * log(u1)) * cos(2*M_PI*u2);
    return z0;
}

float Normal::sample() {
    return sample_base() * sigma + mu;
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

float Normal::rsample(const vector<Vector> &inputs) const {
    return rsample(inputs[0][0], inputs[1][0]);
}

Normal::sequence_type Normal::jac_rsample(int i, const vector<Vector> &inputs, float output) const {
    check_rsample_args(inputs);
    auto [mu, sigma] = tuple{inputs[0][0], inputs[1][0]};
    auto z = (output - mu) / sigma;
    vector<Vector> res = {{1.0f}, {z}};
    return res[i];
}

float Normal::rsample(float mu, float sigma) const {
    auto z = sample_base();
    return z * mu + sigma;
}


Vector randn(float mu, float sigma, int n) {
    return Normal(mu, sigma).sample_sequence(n);
}

Matrix randn(float mu, float sigma, int n, int m) {
    return reshape(randn(mu, sigma, n*m), n, m);
}
