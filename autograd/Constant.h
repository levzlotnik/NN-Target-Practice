//
// Created by LevZ on 6/27/2020.
//

#ifndef TARGETPRACTICE_CONSTANT_H
#define TARGETPRACTICE_CONSTANT_H


#include "VariableBase.h"

namespace autograd {
    template<typename T>
    class Constant : public VariableBase<T> {
    public:
        Constant(const string &name, const Tensor<T> &data) : VariableBase<T>(name, data, false) {}

        void add_dependency(const Variable <T> &dep) override {
            throw runtime_error("A constant cannot be a dependent variable.");
        }

        static Variable <T> make(const string &name, const Tensor<T> &data) {
            return Variable<T>{new Constant{name, data}};
        }

    private:
        string node_style_graphviz() override {
            return "shape=box style=\"filled\"";
        }

    protected:
        void backward(VariableBase <T> *dependee, bool recursive) override {
            // Do nothing - this is a constant.
        }
    };

}
#endif //TARGETPRACTICE_CONSTANT_H
