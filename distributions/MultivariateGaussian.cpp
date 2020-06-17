//
// Created by LevZ on 6/15/2020.
//

#include "MultivariateGaussian.h"

#include <utility>
#include <sstream>

float MultivariateGaussian::logp(Vector val) {
    auto delta = (val - mu) / sigma;
    return float(-0.5f)*(delta*delta).sum();
}

Vector MultivariateGaussian::random() {
    Vector res(k);
    for (int i = 0; i < k; ++i)
        res[i] = generators[i].random();
    return res;
}

MultivariateGaussian::MultivariateGaussian(Vector mu, Vector sigma) :
    MultivariateDistribution(mu.n),
    mu(std::move(mu)), sigma(std::move(sigma)){
    if (mu.n != sigma.n)
        throw runtime_error("Shape mismatch: " + to_string(mu.n) + ", " + to_string(sigma.n));
    for (int i = 0; i < k; ++i)
        generators.emplace_back(this->mu[i], this->sigma[i]);
}

ostream &MultivariateGaussian::print(ostream &os, string indent, bool deep) const {
    ostringstream ss;
    ss << mu << sigma;
    bool endlines = ss.str().size() > 50;
    string endline_ = (endlines ? "\n" : ""), indent_ = (endlines ? indent : "");
    os << "MultivariateGaussian(" << endline_;
    os << indent_ << "mu=" << mu << ", " << endline_;
    os << indent_ << "sigma=" << sigma << endline_;
    return os << indent_  << ")";
}

MultivariateGaussian *MultivariateGaussian::clone() const {
    return new MultivariateGaussian(*this);
}
