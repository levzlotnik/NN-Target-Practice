//
// Created by LevZ on 6/16/2020.
//

#include "Variable.h"
#include "Functor.h"

void Variable::add_dependency(Variable *depend) {
    if(!depend) {
        warning::warn("Adding null dependency is ignored.");
        return;
    }
    dependencies.push_back(depend);
    depend->dependees.push_back(this);
}


Vector &Variable::get_data() {
    return data;
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
    for (auto dep: dependencies)
        dep->check_graph_integrity(visited);
    visited.erase(this);
}

Vector &Variable::grad() {
    return *ptr_grad_data;
}

bool Variable::is_root() const {
    return dependees.empty();
}

void Variable::backward() {
    if(!is_root())
        warning::warn("Calling '.backward()' on a non-root variable will force the gradients to"
                      "flow from the middle of the graph, and may produce unexpected results. "
                      "Avoid it unless you really know what you're doing.");
    backward(nullptr, true);
}


