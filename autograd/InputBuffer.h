//
// Created by LevZ on 6/26/2020.
//

#ifndef TARGETPRACTICE_INPUTBUFFER_H
#define TARGETPRACTICE_INPUTBUFFER_H


#include <utility>

#include "Variable.h"

class InputBuffer : public Variable {
public:
    Vector forward() override;
    static shared_ptr<Variable> make(string name, Vector data){
        return shared_ptr<Variable>(new InputBuffer(std::move(name), std::move(data)));
    }

    void add_dependency(const shared_ptr<Variable> &dep) override;

    bool is_input_buffer() const final { return true; }

protected:
    InputBuffer(string name, Vector data) : Variable(std::move(name), std::move(data), false) {}
    void backward(Variable *dependee, bool recursive) override;
};


#endif //TARGETPRACTICE_INPUTBUFFER_H
