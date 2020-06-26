//
// Created by LevZ on 6/15/2020.
//

#include "RandomVariable.h"

#include <utility>

RandomVariable::RandomVariable(string name, const Distribution &dist, bool requires_grad) :
    Variable(std::move(name), Vector(dist.sample_shape), requires_grad),
    dist(dist.clone()) { }

RandomVariable::RandomVariable(string name, const UnivariateDistribution &dist, bool requires_grad) :
        RandomVariable(std::move(name), EncapsulateUnivariate{dist}, requires_grad) {}

Vector RandomVariable::sample() {
    return dist->sample();
}

Matrix RandomVariable::sample_sequence(int n) {
    return dist->sample_sequence(n);
}

ostream &operator<<(ostream &os, const RandomVariable& rv) {
    return os << "{ " << rv.name << " ~ " << rv.dist << " }";
}

float RandomVariable::logp(Vector v) {
    return dist->logp(std::move(v));
}

Vector RandomVariable::forward() {
    if (is_leaf())
        _data = dist->sample();
    else
        _data = dist->rsample(get_args());
    return _data;
}

void RandomVariable::backward(Variable *dependee, bool recursive) {
    if (unvisited_dependees.at(dependee) <= 0)
        throw runtime_error("WTF... Did you forget to call '.prepare_backward()'? "
                            "Maybe '.check_graph_integrity()'?");

    unvisited_dependees[dependee]--;
    if (!grad_accumulation_complete())
        return;
    auto args = get_args();
    for (int i=0; i < dependencies.size(); ++i) {
        auto jac = dist->jac_rsample(i, args, this->_data);
        auto dep_grad = matmul(grad(), jac);
        dependencies[i]->accumulate_grad(dep_grad);
        if (recursive)
            dependencies[i]->backward(this, true);
    }
}


vector<Vector> RandomVariable::get_args() {
    vector<Vector> args;
    for (auto dep: dependencies)
        args.emplace_back(dep->data());
    return args;
}
