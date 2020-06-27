//
// Created by LevZ on 6/16/2020.
//

#include "VariableBase.h"
#include "../Functor.h"

void VariableBase::add_dependency(const Variable &dep) {
    if(!dep) {
        warning::warn("Adding null dependency is ignored.");
        return;
    }
    dependencies.push_back(dep);
    dep->dependees.push_back(this);
}


Vector &VariableBase::data() {
    return _data;
}

bool VariableBase::is_leaf() const {
    return dependencies.empty();
}

void VariableBase::check_graph_integrity() {
    unordered_set<VariableBase*> visited;
    check_graph_integrity(visited);
}

void VariableBase::check_graph_integrity(unordered_set<VariableBase *>& visited) {
    if (visited.count(this) > 0)
        throw runtime_error("Graph contains cycles. Redefine the graph to not contain cycles.");
    visited.insert(this);
    for (const auto& dep: dependencies)
        dep->check_graph_integrity(visited);
    visited.erase(this);
}

Vector &VariableBase::grad() {
    return _grad;
}

bool VariableBase::is_root() const {
    return dependees.empty();
}

void VariableBase::backward() {
    if(!is_root())
        warning::warn("Calling '.backward()' on a non-root variable will force the gradients to"
                      "flow from the middle of the graph, and may produce unexpected results. "
                      "Avoid it unless you really know what you're doing.");
    prepare_backward();
    grad().fill_(1);
    backward(nullptr, true);
}


void VariableBase::accumulate_grad(const Vector &grad) {
    if (requires_grad)
        _grad += grad;
}

void VariableBase::prepare_backward() {
    unvisited_dependees.clear();
    for (auto dependeePtr: dependees)
        unvisited_dependees[dependeePtr]++;
    for (const auto& dep: dependencies)
        dep->prepare_backward();
}


bool VariableBase::grad_accumulation_complete() const {
    for (auto [dep_ptr, required_visits] : unvisited_dependees)
        if (required_visits > 0)
            return false;
    return true;
}


void VariableBase::zero_grad(bool recursive) {
    if(!requires_grad)
        return;
    _grad.fill_(0);
    if (recursive)
        for (const auto& dep: dependencies)
            dep->zero_grad(true);
}


Vector VariableBase::forward_recursive() {
    for (const auto& dep: dependencies)
        dep->forward_recursive();
    return forward();
}
