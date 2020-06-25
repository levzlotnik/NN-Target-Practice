//
// Created by LevZ on 6/15/2020.
//

#ifndef TARGETPRACTICE_UNIFORM_H
#define TARGETPRACTICE_UNIFORM_H

#include "DistributionBase.h"
#include <random>

using namespace std;

class Uniform : public UnivariateDistribution {
private:
    default_random_engine generator;

    static void check_rsample_args(const vector<Vector>& inputs) {
        if (inputs.size() != 2 || inputs[0].shape() != 1 || inputs[0].shape() != inputs[1].shape())
            throw invalid_argument("Uniform.rsample accepts two arguments of shape=1: {lower, upper}.");
    }

public:

    float lower, upper;
    Uniform(float lower, float upper);

    float logcdf(float val);
    float logp(sample_type val) override;

    sample_type sample() override;
    sample_type sample() const;


    ostream &print(ostream &os, string indent, bool deep) const override;

    sample_type rsample(const vector<Vector> &inputs) const override;
    sample_type rsample(float lower, float upper) const;

    sequence_type jac_rsample(int i, const vector<Vector> &inputs, sample_type output) const override;

    [[nodiscard]] Uniform *clone() const override;
};


#endif //TARGETPRACTICE_UNIFORM_H
