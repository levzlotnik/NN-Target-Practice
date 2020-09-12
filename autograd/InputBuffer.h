//
// Created by LevZ on 6/26/2020.
//

#ifndef TARGETPRACTICE_INPUTBUFFER_H
#define TARGETPRACTICE_INPUTBUFFER_H


#include <utility>

#include "VariableBase.h"

namespace autograd {
    template<typename T>
    class InputBuffer : public VariableBase<T> {
    public:
        Tensor<T> forward() override;

        static Variable<T> make(string name, const Tensor<T>& data) {
            return Variable{new InputBuffer(std::move(name), std::move(data))};
        }

        void add_dependency(const Variable<T> &dep) override;

        bool is_input_buffer() const final { return true; }

    private:
        string node_style_graphviz() override;

    protected:
        InputBuffer(string name, const Tensor<T>& data) : VariableBase<T>(std::move(name), data, false) {}

        void backward(VariableBase *dependee, bool recursive) override;
    };

}
#endif //TARGETPRACTICE_INPUTBUFFER_H
