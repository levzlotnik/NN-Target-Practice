//
// Created by LevZ on 6/16/2020.
//

#include "Variable.h"
#include "Functor.h"

void Variable::add_dependency(const shared_ptr<Variable>& dep) {
    if(!dep) {
        warning::warn("Adding null dependency is ignored.");
        return;
    }
    dependencies.push_back(dep);
    dep->dependees.push_back(this);
}


Vector &Variable::data() {
    return _data;
}

bool Variable::is_leaf() const {
    return dependencies.empty();
}

void Variable::check_graph_integrity() {
    unordered_set<Variable*> visited;
    check_graph_integrity(visited);
}

void Variable::check_graph_integrity(unordered_set<Variable *>& visited) {
    if (visited.count(this) > 0)
        throw runtime_error("Graph contains cycles. Redefine the graph to not contain cycles.");
    visited.insert(this);
    for (const auto& dep: dependencies)
        dep->check_graph_integrity(visited);
    visited.erase(this);
}

Vector &Variable::grad() {
    return _grad;
}

bool Variable::is_root() const {
    return dependees.empty();
}

void Variable::backward() {
    if(!is_root())
        warning::warn("Calling '.backward()' on a non-root variable will force the gradients to"
                      "flow from the middle of the graph, and may produce unexpected results. "
                      "Avoid it unless you really know what you're doing.");
    prepare_backward();
    backward(nullptr, true);
}


void Variable::accumulate_grad(const Vector &grad) {
    if (requires_grad)
        _grad += grad;
}

void Variable::prepare_backward() {
    unvisited_dependees.clear();
    for (auto dependeePtr: dependees)
        unvisited_dependees[dependeePtr]++;
    for (const auto& dep: dependencies)
        dep->prepare_backward();
}


bool Variable::grad_accumulation_complete() const {
    for (auto [dep_ptr, required_visits] : unvisited_dependees)
        if (required_visits > 0)
            return false;
    return true;
}


void Variable::zero_grad(bool recursive) {
    _grad.apply_([](float& x) { return 0; });
    if (recursive)
        for (const auto& dep: dependencies)
            dep->zero_grad(true);
}


Vector Variable::forward_recursive() {
    for (const auto& dep: dependencies)
        dep->forward_recursive();
    return forward();
}
