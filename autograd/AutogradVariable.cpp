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

void AutogradVariable::backward(Variable *dependee, bool recursive) {
    if (unvisited_dependees.at(dependee) <= 0)
        throw runtime_error("WTF... Did you forget to call '.prepare_backward()'? "
                            "Maybe '.check_graph_integrity()'?");

    unvisited_dependees[dependee]--;
    if (!grad_accumulation_complete())
        return;
    auto args = get_args();
    for(int i=0; i < dependencies.size(); ++i) {
        auto jac = source_functor_ptr->jac(i, args, this->data);
        auto dep_grad = matmul(this->grad(), jac);
        dependencies[i]->accumulate_grad(dep_grad);
        if (recursive)
            dependencies[i]->backward(this, true);
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

void AutogradVariable::prepare_backward() {
    unvisited_dependees.clear();
    for (auto dependeePtr: dependees)
        unvisited_dependees[dependeePtr]++;
    for (auto dep: dependencies)
        dep->prepare_backward();
}

bool AutogradVariable::is_root() const {
    return Variable::is_root() && data.n == 1;
}

AutogradVariable::AutogradVariable(string name, const Functor &source_functor, bool requires_grad) :
        Variable(std::move(name), Vector(source_functor.output_shape), requires_grad),
        source_functor_ptr(source_functor.clone()) {}

bool AutogradVariable::grad_accumulation_complete() {
    for (auto [dep_ptr, required_visits] : unvisited_dependees)
        if (required_visits > 0)
            return false;
    return true;
}
