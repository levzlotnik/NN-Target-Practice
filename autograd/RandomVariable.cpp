//
// Created by LevZ on 6/15/2020.
//

#include "RandomVariable.h"

#include <utility>

RandomVariable::RandomVariable(string name, const Distribution& dist) :
    name(std::move(name)),
    dist(dist.clone()) {

}

RandomVariable::RandomVariable(string name, const UnivariateDistribution &dist) : name(std::move(name)),
    dist(new EncapsulateUnivariate(dist)){

}

Vector RandomVariable::sample() {
    return dist->sample();
}

Matrix RandomVariable::sample_n(int n) {
    return dist->sample_sequence(n);
}

ostream &operator<<(ostream &os, const RandomVariable& rv) {
    return os << "[ " << rv.name << " ~ " << rv.dist << " ]";
}

float RandomVariable::logp(Vector v) {
    return dist->logp(std::move(v));
}

void RandomVariable::accumulate_grad(const Vector &jac) {

}

void RandomVariable::forward() {

}

void RandomVariable::backward(Variable *dependee, bool recursive) {

}

void RandomVariable::zero_grad(bool recursive) {

}
