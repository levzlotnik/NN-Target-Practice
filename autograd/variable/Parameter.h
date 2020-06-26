//
// Created by LevZ on 6/26/2020.
//

#ifndef TARGETPRACTICE_PARAMETER_H
#define TARGETPRACTICE_PARAMETER_H

#include <utility>

#include "Variable.h"

// Represents a leaf variable with data only.
class Parameter : public Variable {
public:
    Vector forward() override;

    void add_dependency(const shared_ptr<Variable>& dep) override;

    inline bool is_param() const final { return true; }

    static shared_ptr<Variable> make(string name, Vector data, bool requires_grad=true) {
        return shared_ptr<Variable>(new Parameter(std::move(name), std::move(data), requires_grad));
    }

protected:
    using Variable::Variable;
    void backward(Variable *dependee, bool recursive) override;
};


#endif //TARGETPRACTICE_PARAMETER_H
