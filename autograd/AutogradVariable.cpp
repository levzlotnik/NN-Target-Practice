//
// Created by LevZ on 6/17/2020.
//

#include "AutogradVariable.h"

void AutogradVariable::accumulate_grad(const Vector &jac) {
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

void AutogradVariable::backward(const Vector& current_grad, bool recursive) {
    auto args = get_args();
    for(int i=0; i < dependencies.size(); ++i){
        auto jac = source_functor_ptr->jac(i, args, this->data);
        auto dep_grad = matmul(current_grad, *jac);
        dependencies[i]->accumulate_grad(dep_grad);
         // TODO - this is not good. For instance - if we have a skip-connection:
         //  h_{n+2} = h_{n} + h_{n+1}(h_{n})
         //  we have 2 gradient paths: (1) h_{n+2}->h_{n}; (2) h_{n+2}->h_{n+1}->h_{n}.
         //  After (1) finishes - this recursive call makes him call `.backward()` instantly, without waiting
         //  for (2) to finish. We need a graph traversal that allows to wait until all gradients got to the
         //  variable and only then continue the recursive call. For instance - we can have a constant
         //  unordered_set<Variable*> dependees, and each call to backward that gets to some variable it removes a
         //  dependee (resets the set after calling `.zero_grad()`). This allows us to accumulate the correct gradient
         //  before proceeding to go to next nodes.
        if (recursive)
            dependencies[i]->backward(dep_grad, true);
    }
}

void AutogradVariable::zero_grad(bool recursive) {
    if (requires_jac_init)
        warning::warn("Warning: zeroing jacobian without initializing it is will have no effect.");
    this->jac.apply_([](float& x) {return 0;});
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
