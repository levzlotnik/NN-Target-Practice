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
    return dist->random();
}

Matrix RandomVariable::sample_n(int n) {
    return dist->random_sequence(n);
}

ostream &operator<<(ostream &os, const RandomVariable& rv) {
    return os << "[ " << rv.name << " ~ " << rv.dist << " ]";
}

float RandomVariable::logp(Vector v) {
    return dist->logp(std::move(v));
}
