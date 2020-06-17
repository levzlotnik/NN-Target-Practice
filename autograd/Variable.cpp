//
// Created by LevZ on 6/16/2020.
//

#include "Variable.h"
#include "VectorFunction.h"

void Variable::add_dependency(Variable *depend) {
    if(!depend) {
        warning::warn("Adding null dependency is ignored.");
        return;
    }
    dependencies.push_back(depend);
}

void Variable::accumulate_jac(const Matrix& jac) {
    if(this->ptrJac == nullptr)
        this->ptrJac = make_unique<Matrix>(jac);
    else
        *this->ptrJac += jac;

}

void Variable::backward(bool recursive) {
    if (is_leaf())
        return;
    // gather arguments:
    vector<Vector> args;
    for (auto p: dependencies)
        args.push_back(p->get_data());
    // calculate jacobians and accumulate them:
    for (int i = 0; i < args.size(); ++i) {
        dependencies[i]->accumulate_jac(functor->jac(i, args, this->data));
        if (recursive) dependencies[i]->backward(true);
    }
}

void Variable::forward() {
    if (is_leaf())
        return;
    // gather arguments:
    vector<Vector> args;
    for (auto p: dependencies)
        args.push_back(p->get_data());

    this->data = (*functor)(args);
}

void Variable::zero_grad() {
    this->ptrJac->apply_([](float& x){return 0;});
}

Vector &Variable::get_data() {
    return data;
}

bool Variable::is_leaf() {
    return dependencies.empty();
}

void Variable::set_functor(const VectorFunction &vectorFunction) {
    functor = vectorFunction.clone();
}

Variable::~Variable() {
    delete functor;
}
