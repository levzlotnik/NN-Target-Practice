//
// Created by LevZ on 6/15/2020.
//

#ifndef BLAS_MULTIVARIATECONSTANT_H
#define BLAS_MULTIVARIATECONSTANT_H

#include "DistributionBase.h"
#include "../Vector.h"

class MultivariateConstant : public MultivariateDistribution {
protected:
    Vector v;

public:
    explicit MultivariateConstant(Vector v);

    float logp(T val) override;

    T random() override;

    ostream &print(ostream &os, string indent, bool deep) const override;

    MultivariateConstant *clone() const override;
};

class Data : public MultivariateConstant {
public:
    using MultivariateConstant::MultivariateConstant;
    ostream &print(ostream &os, string indent, bool deep) const override;
};

#endif //BLAS_MULTIVARIATECONSTANT_H
