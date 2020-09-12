//
// Created by LevZ on 6/26/2020.
//

#ifndef TARGETPRACTICE_PARAMETER_H
#define TARGETPRACTICE_PARAMETER_H

#include <utility>

#include "VariableBase.h"

namespace autograd {
// Represents a leaf variable with data only.
    template<typename T>
    class Parameter : public VariableBase<T> {
    public:
        Tensor<T> forward() override;

        void add_dependency(const Variable<T> &dep) override;

        inline bool is_param() const final { return true; }

        static Variable<T> make(const string &name, Tensor<T> data, bool requires_grad = true) {
            return Variable<T>{new Parameter(name, std::move(data), requires_grad)};
        }

    private:
        string node_style_graphviz() override;

    protected:
        using VariableBase<T>::VariableBase;

        void backward(VariableBase<T> *dependee, bool recursive) override;
    };
}

#endif //TARGETPRACTICE_PARAMETER_H
