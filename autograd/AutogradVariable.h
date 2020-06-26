//
// Created by LevZ on 6/17/2020.
//

#ifndef TARGETPRACTICE_AUTOGRADVARIABLE_H
#define TARGETPRACTICE_AUTOGRADVARIABLE_H


#include "Variable.h"
#include "Functor.h"

class AutogradVariable : public Variable {
private:
    // The functor that creates the data for the current variable.
    shared_ptr<Functor> source_functor_ptr;
    vector<Vector> get_args();
    void backward(Variable *dependee, bool recursive) override;

public:
    using Variable::Variable;

    AutogradVariable(string name, const Functor& source_functor, bool requires_grad=true);

    Vector forward() override;

    // Returns true if this variable has no dependencies.
    // Autograd variable cannot be a root if its' shape isn't 1.
    bool is_root() const override;

};

using Deterministic = AutogradVariable;

#endif //TARGETPRACTICE_AUTOGRADVARIABLE_H
