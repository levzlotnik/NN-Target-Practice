//
// Created by LevZ on 6/15/2020.
//

#ifndef TARGETPRACTICE_RANDOMVARIABLE_H
#define TARGETPRACTICE_RANDOMVARIABLE_H

#include <string>
#include <utility>
#include "../../distributions/DistributionBase.h"
#include "../../BLAS/BLAS.h"
#include "Variable.h"

using namespace std;

class RandomVariable : public Variable {
private:
    shared_ptr<Distribution> dist;
    vector<Vector> get_args();

    RandomVariable(string name, const Distribution &dist, bool requires_grad=true);

    RandomVariable(string name, const UnivariateDistribution &dist, bool requires_grad=true);
public:


    Vector forward() override;

    void backward(Variable *dependee, bool recursive) override;

    friend ostream& operator <<(ostream& os, const RandomVariable& rv);

    Vector sample();
    Matrix sample_sequence(int n);

    static shared_ptr<Variable> make(string name, const Distribution &dist, bool requires_grad=true) {
        return shared_ptr<Variable>(new RandomVariable(std::move(name), dist, requires_grad));
    }

    static shared_ptr<Variable> make(string name, const UnivariateDistribution &dist, bool requires_grad=true) {
        return shared_ptr<Variable>(new RandomVariable(std::move(name), dist, requires_grad));
    }

    float logp(Vector v);
};


#endif //TARGETPRACTICE_RANDOMVARIABLE_H
