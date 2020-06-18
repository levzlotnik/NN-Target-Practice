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
}


Vector &Variable::get_data() {
    return data;
}

bool Variable::is_leaf() {
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


