//
// Created by LevZ on 6/26/2020.
//

#include "InputBuffer.h"

#include <utility>

Vector InputBuffer::forward() {
    return _data;
}

void InputBuffer::backward(Variable *dependee, bool recursive) {
    // Do nothing - leaf variable.
}

void InputBuffer::add_dependency(const shared_ptr<Variable> &dep) {
    throw runtime_error("InputBuffer is an independent variable, adding dependencies to it will have no effect.");
}
