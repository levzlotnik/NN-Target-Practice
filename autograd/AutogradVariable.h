//
// Created by LevZ on 6/17/2020.
//

#ifndef TARGETPRACTICE_AUTOGRADVARIABLE_H
#define TARGETPRACTICE_AUTOGRADVARIABLE_H


#include "VariableBase.h"
#include "Functor.h"
namespace autograd {

    template<typename T>
    class AutogradVariable : public VariableBase<T> {
    private:
        // The functor that creates the data for the current variable.
        shared_ptr<Functor<T>> source_functor_ptr;

        vector<Tensor<T>> get_args();

        void backward(VariableBase<T> *dependee, bool recursive) override;

        AutogradVariable(const string &name, const Functor<T> &source_functor, bool requires_grad = true);

    public:
        Tensor<T> forward() override;

        // Returns true if this variable has no dependencies.
        // Autograd variable cannot be a root if its' shape isn't 1.
        bool is_root() const override;

        static Variable<T> make(const string &name, const Functor<T> &source_functor, bool requires_grad = true) {
            Variable<T> res{new AutogradVariable(name, source_functor, requires_grad)};
            return res;
        }

    };

    template<typename T>
    using Deterministic = AutogradVariable<T>;
}

#endif //TARGETPRACTICE_AUTOGRADVARIABLE_H
