//
// Created by LevZ on 6/26/2020.
//

#ifndef TARGETPRACTICE_INPUTBUFFER_H
#define TARGETPRACTICE_INPUTBUFFER_H


#include <utility>

#include "Variable.h"

class InputBuffer : public Variable {
public:
    InputBuffer(string name, Vector data) : Variable(std::move(name), std::move(data), false) {}
    Vector forward() override;

protected:
    void backward(Variable *dependee, bool recursive) override;
};


#endif //TARGETPRACTICE_INPUTBUFFER_H
