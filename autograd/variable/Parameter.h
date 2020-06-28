//
// Created by LevZ on 6/26/2020.
//

#ifndef TARGETPRACTICE_PARAMETER_H
#define TARGETPRACTICE_PARAMETER_H

#include <utility>

#include "VariableBase.h"

// Represents a leaf variable with data only.
class Parameter : public VariableBase {
public:
    Vector forward() override;

    void add_dependency(const Variable &dep) override;

    inline bool is_param() const final { return true; }

    static Variable make(string name, Vector data, bool requires_grad=true) {
        return Variable{new Parameter(std::move(name), std::move(data), requires_grad)};
    }

private:
    string node_style_graphviz() override;

protected:
    using VariableBase::VariableBase;
    void backward(VariableBase *dependee, bool recursive) override;
};


#endif //TARGETPRACTICE_PARAMETER_H
