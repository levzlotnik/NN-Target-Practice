//
// Created by LevZ on 6/17/2020.
//

#ifndef TARGETPRACTICE_AUTOGRADVARIABLE_H
#define TARGETPRACTICE_AUTOGRADVARIABLE_H


#include "VariableBase.h"
#include "../Functor.h"

class AutogradVariable : public VariableBase {
private:
    // The functor that creates the data for the current variable.
    shared_ptr<Functor> source_functor_ptr;
    vector<Vector> get_args();
    void backward(VariableBase *dependee, bool recursive) override;

    AutogradVariable(string name, const Functor& source_functor, bool requires_grad=true);

public:
    Vector forward() override;

    // Returns true if this variable has no dependencies.
    // Autograd variable cannot be a root if its' shape isn't 1.
    bool is_root() const override;

    static Variable make(const string& name, const Functor& source_functor, bool requires_grad=true) {
        Variable res{new AutogradVariable(name, source_functor, requires_grad)};
        return res;
    }

};

using Deterministic = AutogradVariable;

#endif //TARGETPRACTICE_AUTOGRADVARIABLE_H
