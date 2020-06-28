//
// Created by LevZ on 6/15/2020.
//

#ifndef TARGETPRACTICE_RANDOMVARIABLE_H
#define TARGETPRACTICE_RANDOMVARIABLE_H

#include <string>
#include <utility>
#include "../../distributions/DistributionBase.h"
#include "../../BLAS/BLAS.h"
#include "VariableBase.h"

using namespace std;

class RandomVariable : public VariableBase {
private:
    shared_ptr<Distribution> dist;
    vector<Vector> get_args();

    RandomVariable(string name, const Distribution &dist, bool requires_grad=true);

    RandomVariable(string name, const UnivariateDistribution &dist, bool requires_grad=true);
public:


    Vector forward() override;

    void backward(VariableBase *dependee, bool recursive) override;

    friend ostream& operator <<(ostream& os, const RandomVariable& rv);

    Vector sample();
    Matrix sample_sequence(int n);

    static Variable make(string name, const Distribution &dist, bool requires_grad=true) {
        return Variable{new RandomVariable(std::move(name), dist, requires_grad)};
    }

    static shared_ptr<VariableBase> make(string name, const UnivariateDistribution &dist, bool requires_grad=true) {
        return shared_ptr<VariableBase>(new RandomVariable(std::move(name), dist, requires_grad));
    }

    float logp(Vector v);
};

void VariableBase::remove_dependency(const Variable &dep) {
    auto& d_dependees = dep->dependees;
    d_dependees.erase(std::remove(d_dependees.begin(), d_dependees.end(), this), d_dependees.end());
    dependencies.erase(std::remove(dependencies.begin(), dependencies.end(), dep), dependencies.end());
}

VariableBase::~VariableBase() {
    for (auto dep: dependencies){
        auto& d_dependees = dep->dependees;
        d_dependees.erase(std::remove(d_dependees.begin(), d_dependees.end(), this), d_dependees.end());
    }
}


#endif //TARGETPRACTICE_RANDOMVARIABLE_H
