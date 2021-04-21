//
// Created by LevZ on 6/26/2020.
//

#ifndef TARGETPRACTICE_PARAMETER_H
#define TARGETPRACTICE_PARAMETER_H

#include <utility>

#include "VariableBase.h"

namespace autograd {
// Represents a leaf variable with data only.
template <typename T>
class Parameter : public VariableBase<T> {
   public:
    void add_dependency(const Variable<T>& dep) override {
        throw runtime_error(
            "Parameter is an independent variable, "
            "adding dependencies to it will have no effect.");
    }

    inline bool is_param() const final { return true; }

    static Variable<T> make(const string& name, Tensor<T> t,
                            bool requires_grad = true) {
        return Variable<T>{new Parameter(name, std::move(t), requires_grad)};
    }

   private:
    string node_style_graphviz() override {
        return "shape=box style=\"rounded\"";
    }

   protected:
    using VariableBase<T>::VariableBase;

    void backward(VariableBase<T>* dependee, bool recursive) override {
        // Do nothing - this is leaf variable.
    }
};
}  // namespace autograd

#endif  // TARGETPRACTICE_PARAMETER_H
