//
// Created by LevZ on 6/15/2020.
//

#ifndef TARGETPRACTICE_MULTIVARIATEGAUSSIAN_H
#define TARGETPRACTICE_MULTIVARIATEGAUSSIAN_H

#include <vector>
#include "Normal.h"

using namespace std;

class MultivariateGaussian : public MultivariateDistribution {
private:
    vector<Normal> generators;
    static void check_rsample_args(const vector<Vector>& inputs) {
        if (inputs.size() != 2 || inputs[0].shape() != inputs[1].shape())
            throw invalid_argument("MultivariateGaussian.rsample expects two arguments of the same shape: {mu, sigma}.");
    }
public:
    Vector mu, sigma;

    MultivariateGaussian(Vector mu, Vector sigma);
    explicit MultivariateGaussian(int shape) : MultivariateDistribution(shape) {}

    float logp(sample_type val) override;

    sample_type sample() override;

    ostream &print(ostream &os, string indent, bool deep) const override;

    sample_type rsample(const vector<Vector> &inputs) const override;

    sequence_type jac_rsample(int i, const vector<Vector> &inputs, sample_type output) const override;

    MultivariateGaussian *clone() const override;
};


#endif //TARGETPRACTICE_MULTIVARIATEGAUSSIAN_H
