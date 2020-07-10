//
// Created by LevZ on 6/26/2020.
//

#include "Parameter.h"

Vector Parameter::forward() {
    // Do nothing - this is leaf variable.
    return _data;
}

void Parameter::backward(VariableBase *dependee, bool recursive) {
    // Do nothing - this is leaf variable.
}

void Parameter::add_dependency(const Variable &dep) {
    throw runtime_error("Parameter is an independent variable, adding dependencies to it will have no effect.");
}

string Parameter::node_style_graphviz() {
    return "shape=box style=\"rounded\"";
}