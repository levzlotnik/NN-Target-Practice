//
// Created by LevZ on 6/15/2020.
//

#include "MultivariateGaussian.h"
#include "DistributionBase.h"

#include <utility>
#include <sstream>

float MultivariateGaussian::logp(Vector val) {
    auto delta = (val - mu) / sigma;
    return float(-0.5f)*(delta*delta).sum();
}

Vector MultivariateGaussian::sample() {
    Vector res(sample_shape);
    for (int i = 0; i < sample_shape; ++i)
        res[i] = generators[i].sample();
    return res;
}

MultivariateGaussian::MultivariateGaussian(Vector mu, Vector sigma) :
    MultivariateDistribution(mu.n),
    mu(std::move(mu)), sigma(std::move(sigma)){
    if (mu.n != sigma.n)
        throw runtime_error("Shape mismatch: " + to_string(mu.n) + ", " + to_string(sigma.n));
    for (int i = 0; i < sample_shape; ++i)
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

Vector MultivariateGaussian::rsample(const vector<Vector> &inputs) const {
    check_rsample_args(inputs);
    auto mu_ = inputs[0], sigma_ = inputs[1];
    Normal gen;
    Vector res(mu_.n);
    for (int i=0; i < mu_.n; ++i){
        res[i] = gen.rsample(mu_[i], sigma_[i]);
    }
    return res;
}

MultivariateGaussian::sequence_type
MultivariateGaussian::jac_rsample(int i, const vector<Vector> &inputs, Vector output) const {
    check_rsample_args(inputs);
    if (i < 0 || i > 1)
        throw out_of_range("Only two arguments available.");
    auto [mu_, sigma_] = tuple{inputs[0], inputs[1]};
    return i == 0 ?
        Matrix::diag(Vector::ones_like(mu_), true) :
        Matrix::diag((output - mu_) / sigma_, true);
}
