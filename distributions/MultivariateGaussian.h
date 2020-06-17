//
// Created by LevZ on 6/15/2020.
//

#ifndef BLAS_MULTIVARIATEGAUSSIAN_H
#define BLAS_MULTIVARIATEGAUSSIAN_H

#include <vector>
#include "Normal.h"

using namespace std;

class MultivariateGaussian : public virtual MultivariateDistribution {
private:
    vector<Normal> generators;
public:
    Vector mu, sigma;

    MultivariateGaussian(Vector mu, Vector sigma);

    float logp(T val) override;

    T random() override;

    ostream &print(ostream &os, string indent, bool deep) const override;

    MultivariateGaussian *clone() const override;
};


#endif //BLAS_MULTIVARIATEGAUSSIAN_H