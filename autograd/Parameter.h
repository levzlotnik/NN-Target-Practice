//
// Created by LevZ on 6/26/2020.
//

#ifndef TARGETPRACTICE_PARAMETER_H
#define TARGETPRACTICE_PARAMETER_H

#include "Variable.h"

// Represents a leaf variable with data only.
class Parameter : public Variable {
public:
    using Variable::Variable;

    Vector forward() override;

    void add_dependency(Variable *dep) override;

protected:
    void backward(Variable *dependee, bool recursive) override;
};


#endif //TARGETPRACTICE_PARAMETER_H
