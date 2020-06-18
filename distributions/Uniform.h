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

public:

    float lower, upper;
    Uniform(float lower, float upper);

    float logcdf(float val);
    float logp(T val) override;

    T random() override;

    ostream &print(ostream &os, string indent, bool deep) const override;

    Uniform *clone() const override;
};


#endif //TARGETPRACTICE_UNIFORM_H
