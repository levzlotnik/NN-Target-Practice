//
// Created by LevZ on 6/17/2020.
//

#ifndef TARGETPRACTICE_AUTOGRADVARIABLE_H
#define TARGETPRACTICE_AUTOGRADVARIABLE_H


#include "Variable.h"
#include "Functor.h"

class AutogradVariable : public Variable {
protected:
    // The functor that creates the data for the current variable.
    shared_ptr<Functor> source_functor_ptr;
    vector<Vector> get_args();
    unordered_map<Variable*, int> unvisited_dependees;
    void backward(Variable *dependee, bool recursive) override;

public:
    using Variable::Variable;

    AutogradVariable(string name, const Functor& source_functor, bool requires_grad=true);

    void accumulate_grad(const Vector &grad) override;

    void forward() override;

    void prepare_backward() override;

    void zero_grad(bool recursive) override;

    // Returns true if this variable has no dependencies.
    // Autograd variable cannot be a root if its' shape isn't 1.
    bool is_root() const override;

    bool grad_accumulation_complete();
};


#endif //TARGETPRACTICE_AUTOGRADVARIABLE_H
