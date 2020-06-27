//
// Created by LevZ on 6/26/2020.
//

#ifndef TARGETPRACTICE_INPUTBUFFER_H
#define TARGETPRACTICE_INPUTBUFFER_H


#include <utility>

#include "VariableBase.h"

class InputBuffer : public VariableBase {
public:
    Vector forward() override;
    static Variable make(string name, Vector data){
        return Variable{new InputBuffer(std::move(name), std::move(data))};
    }

    void add_dependency(const Variable &dep) override;

    bool is_input_buffer() const final { return true; }

protected:
    InputBuffer(string name, Vector data) : VariableBase(std::move(name), std::move(data), false) {}
    void backward(VariableBase *dependee, bool recursive) override;
};


#endif //TARGETPRACTICE_INPUTBUFFER_H
