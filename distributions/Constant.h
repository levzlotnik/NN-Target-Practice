//
// Created by LevZ on 6/15/2020.
//

#ifndef TARGETPRACTICE_CONSTANT_H
#define TARGETPRACTICE_CONSTANT_H


#include "DistributionBase.h"

class Constant : public UnivariateDistribution {
private:
    float c;
public:
    ostream &print(ostream &os, string indent, bool deep) const override;

    explicit Constant(float f);

    float logp(T val) override;

    T random() override;

    Constant *clone() const override;
};


#endif //TARGETPRACTICE_CONSTANT_H
