//
// Created by LevZ on 6/26/2020.
//

#include "InputBuffer.h"

#include <utility>

namespace autograd {
    template<typename T>
    Vector InputBuffer<T>::forward() {
        return _data;
    }

    template<typename T>
    void InputBuffer<T>::backward(VariableBase<T> *dependee, bool recursive) {
        // Do nothing - leaf variable.
    }

    template<typename T>
    void InputBuffer<T>::add_dependency(const Variable<T> &dep) {
        throw runtime_error("InputBuffer is an independent variable, adding dependencies to it will have no effect.");
    }

    template<typename T>
    string InputBuffer<T>::node_style_graphviz() {
        return "shape=box style=\"rounded, filled\"";
    }
}