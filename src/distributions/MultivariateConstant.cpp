//
// Created by LevZ on 6/15/2020.
//

#include "../../include/distributions/MultivariateConstant.h"

#include <utility>

float MultivariateConstant::logp(Vector val) {
    return 0;
}

Vector MultivariateConstant::random() {
    return v;
}

MultivariateConstant::MultivariateConstant(Vector v) :
    MultivariateDistribution(v.n),
    v(std::move(v)) {

}

ostream &MultivariateConstant::print(ostream &os, string indent, bool deep) const {
    return os << "MultivariateConstant(" << v << ")";
}

MultivariateConstant *MultivariateConstant::clone() const {
    return new MultivariateConstant(*this);
}

ostream &Data::print(ostream &os, string indent, bool deep) const {
    return os << "Data(" << v << ")";
}
