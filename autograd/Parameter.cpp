//
// Created by LevZ on 6/26/2020.
//

#include "Parameter.h"
namespace autograd {

    template<typename T>
    Tensor<T> Parameter<T>::forward() {
        // Do nothing - this is leaf variable.
        return VariableBase<T>::_data;
    }

    template<typename T>
    void Parameter<T>::backward(VariableBase<T> *dependee, bool recursive) {
        // Do nothing - this is leaf variable.
    }

    template<typename T>
    void Parameter<T>::add_dependency(const Variable<T> &dep) {
        throw runtime_error("Parameter is an independent variable, adding dependencies to it will have no effect.");
    }

    template<typename T>
    string Parameter<T>::node_style_graphviz() {
        return "shape=box style=\"rounded\"";
    }
}