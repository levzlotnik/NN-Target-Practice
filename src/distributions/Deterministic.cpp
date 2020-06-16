//
// Created by LevZ on 6/15/2020.
//

#include "../../include/distributions/Deterministic.h"

float Deterministic::logp(Vector val) {
    return 0;
}

Vector Deterministic::random() {
    return DistributionBase::T(0);
}

ostream &Deterministic::print(ostream &os, string indent, bool deep) const {
    return os;
}

Deterministic * Deterministic::clone() const {
    return nullptr;
}

