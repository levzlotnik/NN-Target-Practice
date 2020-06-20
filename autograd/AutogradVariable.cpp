//
// Created by LevZ on 6/17/2020.
//

#include "AutogradVariable.h"

void AutogradVariable::accumulate_grad(const Vector &grad) {
    if (ptr_grad_data == nullptr)
        ptr_grad_data = shared_ptr<Vector>(grad.clone());
    else
        this->grad() += grad;
}

void AutogradVariable::forward() {
    if (is_leaf())
        return;
    this->data = (*this->source_functor_ptr)(get_args());
}

void AutogradVariable::backward(const Vector& current_grad, bool recursive) {
    auto args = get_args();
    for(int i=0; i < dependencies.size(); ++i){
        auto jac = source_functor_ptr->jac(i, args, this->data);
        auto dep_grad = matmul(current_grad, *jac);
        dependencies[i]->accumulate_grad(dep_grad);
        if (recursive)
            dependencies[i]->backward(dep_grad, true);
    }
}

void AutogradVariable::zero_grad(bool recursive) {
    this->grad().apply_([](float& x) {return 0;});
    if(recursive)
        for(auto dep: dependencies)
            dep->zero_grad(true);
}

vector<Vector> AutogradVariable::get_args() {
    vector<Vector> arg_refs;
    for (auto dep: dependencies)
        arg_refs.emplace_back(dep->get_data());
    return arg_refs;
}
