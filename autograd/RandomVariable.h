//
// Created by LevZ on 6/15/2020.
//

#ifndef BLAS_RANDOMVARIABLE_H
#define BLAS_RANDOMVARIABLE_H

#include <string>
#include "../distributions/DistributionBase.h"
#include "../BLAS/BLAS.h"

using namespace std;

class RandomVariable /* TODO - inherit from variable */ {
private:
    shared_ptr<Distribution> dist;

public:
    string name;

    RandomVariable(string name, const Distribution& dist);
    RandomVariable(string name, const UnivariateDistribution& dist);

    friend ostream& operator <<(ostream& os, const RandomVariable& rv);

    Vector sample();
    Matrix sample_n(int n);

    float logp(Vector v);
};


#endif //BLAS_RANDOMVARIABLE_H