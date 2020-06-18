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
    Uniform g1, g2;
public:
    Normal(float mu, float sigma);

    float logp(float val) override;

    float random() override;

    ostream &print(ostream &os, string indent, bool deep) const override;

    Normal *clone() const override;
};


#endif //TARGETPRACTICENORMAL_H
