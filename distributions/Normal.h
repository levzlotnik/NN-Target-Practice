//
// Created by LevZ on 6/15/2020.
//

#ifndef TARGETPRACTICENORMAL_H
#define TARGETPRACTICENORMAL_H

#include "Uniform.h"

class Normal : public UnivariateDistribution {
public:

    float mu, sigma;
private:
    const Uniform g1, g2;
    float sample_base() const;

    static void check_rsample_args(const vector<Vector>& inputs) {
        if (inputs.size() != 2 || inputs[0].shape() != 1 || inputs[0].shape() != inputs[1].shape())
            throw invalid_argument("Normal.rsample accepts two arguments of shape=1: {mu, sigma}.");
    }
public:
    Normal(float mu, float sigma);
    Normal() : Normal(0, 1) {}

    float logp(float val) override;

    float sample() override;

    sample_type rsample(const vector<Vector> &inputs) const override;
    sample_type rsample(float mu, float sigma) const;

    sequence_type jac_rsample(int i, const vector<Vector> &inputs, sample_type output) const override;

    ostream &print(ostream &os, string indent, bool deep) const override;

    Normal *clone() const override;
};

Vector randn(float mu, float sigma, int n);

Matrix randn(float mu, float sigma, int n, int m);




#endif //TARGETPRACTICENORMAL_H
