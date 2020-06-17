//
// Created by LevZ on 6/17/2020.
//

#include "AutogradVariable.h"

void AutogradVariable::accumulate_jac(const Matrix &jac) {
    if (requires_jac_init)
        this->jac = jac;
    else
        this->jac += jac;
}

void AutogradVariable::forward() {
    if (is_leaf())
        return;
    this->data = (*this->source_functor_ptr)(get_args());
}

void AutogradVariable::backward(bool recursive) {
    auto args = get_args();
    for(int i=0; i < dependencies.size(); ++i){
        dependencies[i]->accumulate_jac(source_functor_ptr->jac(i, args, this->data));
        if (recursive) dependencies[i]->backward(true);
    }
}

void AutogradVariable::zero_jac(bool recursive) {
    if (requires_jac_init)
        warning::warn("Warning: zeroing jacobian without initializing it is will have no effect.");
    this->jac.apply_([](float& x) {return 0;});
    if(recursive)
        for(auto dep: dependencies)
            dep->zero_jac(true);
}

vector<Vector> AutogradVariable::get_args() {
    vector<Vector> arg_refs;
    for (auto dep: dependencies)
        arg_refs.emplace_back(dep->get_data());
    return arg_refs;
}
