//
// Created by LevZ on 6/27/2020.
//

#ifndef TARGETPRACTICE_CONSTANT_H
#define TARGETPRACTICE_CONSTANT_H


#include "VariableBase.h"

class Constant : public VariableBase {
public:
    Constant(const string& name, const Vector& data) : VariableBase(name, data, false) {}

    Vector forward() override {
        return _data;
    }

    void add_dependency(const Variable &dep) override {
        throw runtime_error("A constant cannot be a dependent variable.");
    }

    static Variable make(const string& name, const Vector& data) {
        return Variable{new Constant{name, data}};
    }

protected:
    void backward(VariableBase *dependee, bool recursive) override {
        // Do nothing - this is a constant.
    }
};


#endif //TARGETPRACTICE_CONSTANT_H
