//
// Created by LevZ on 6/15/2020.
//

#ifndef TARGETPRACTICE_RANDOMVARIABLE_H
#define TARGETPRACTICE_RANDOMVARIABLE_H

#include <string>
#include "../distributions/DistributionBase.h"
#include "../BLAS/BLAS.h"

using namespace std;

class RandomVariable : public Variable {
private:
    shared_ptr<Distribution> dist;

public:
    string name;

    RandomVariable(string name, const Distribution& dist);

    RandomVariable(string name, const UnivariateDistribution& dist);

    void accumulate_grad(const Vector &jac) override;

    void forward() override;

    void backward(bool recursive) override;

    void zero_grad(bool recursive) override;

    friend ostream& operator <<(ostream& os, const RandomVariable& rv);

    Vector sample();
    Matrix sample_n(int n);

    float logp(Vector v);
};


#endif //TARGETPRACTICE_RANDOMVARIABLE_H
