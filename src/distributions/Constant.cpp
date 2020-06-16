//
// Created by LevZ on 6/15/2020.
//

#include "../../include/distributions/Constant.h"

float Constant::logp(float val) {
    return 0;
}

float Constant::random() {
    return c;
}

Constant::Constant(float f) : c(f){

}

ostream &Constant::print(ostream &os, string indent, bool deep) const {
    return os << "Constant(c = " << c << ")";
}

Constant *Constant::clone() const {
    return new Constant(*this);
}
