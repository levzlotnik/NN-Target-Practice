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
    bool requires_grad;
    vector<Vector> get_args();

public:
    using Variable::Variable;

    explicit AutogradVariable(const Functor& source_functor, bool requires_grad=true) :
            Variable(Vector(source_functor.output_shape)),
            source_functor_ptr(source_functor.clone()),
            requires_grad(requires_grad){}


    void accumulate_jac(const Matrix &jac) override;

    void forward() override;

    void backward(bool recursive) override;

    void zero_jac(bool recursive) override;
};


#endif //TARGETPRACTICE_AUTOGRADVARIABLE_H
