//
// Created by LevZ on 6/15/2020.
//

#ifndef BLAS_CONSTANT_H
#define BLAS_CONSTANT_H


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


#endif //BLAS_CONSTANT_H
