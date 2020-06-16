//
// Created by LevZ on 6/15/2020.
//

#ifndef BLAS_DETERMINISTIC_H
#define BLAS_DETERMINISTIC_H


#include "DistributionBase.h"
#include "RandomVariable.h"
#include "../VectorFunction.h"
#include <vector>

class Deterministic : public Distribution {
    // NOTE - Deterministic may return a scalar, but only in a 1-element Vector.
private:
    unique_ptr<VectorFunction> functor;
public:
    vector<RandomVariable> arguments;

    Deterministic(const vector<RandomVariable>& args, const VectorFunction& functor);

    float logp(T val) override;

    T random() override;

    ostream &print(ostream &os, string indent, bool deep) const override;

    Deterministic *clone() const override;
};


#endif //BLAS_DETERMINISTIC_H
