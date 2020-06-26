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
    warning::warn("InputBuffer is an independent variable, adding dependencies to it will have no effect.");
}

Vector &InputBuffer::set_data(Vector data) {
    return (this->_data = std::move(data));
}
