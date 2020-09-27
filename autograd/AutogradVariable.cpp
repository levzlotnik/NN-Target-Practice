//
// Created by LevZ on 6/17/2020.
//

#include "AutogradVariable.h"
namespace autograd {
    template<typename T>
    Tensor<T> AutogradVariable<T>::forward() {
        Tensor<T>& out = Variable<T>::data();
        if (!this->is_leaf())
            source_functor_ptr->apply_forward(get_args(), &out);
        return out;
    }

    template<typename T>
    void AutogradVariable<T>::backward(VariableBase<T> *dependee, bool recursive) {
//    cout << name << ".backward()" << endl;
        if (!this->requires_grad)
            return;
        if (dependee) {
            if (this->unvisited_dependees.at(dependee) <= 0)
                throw runtime_error("WTF... Did you forget to call '.prepare_backward()'? "
                                    "Maybe '.check_graph_integrity()'?");

            this->unvisited_dependees[dependee]--;
            if (!this->grad_accumulation_complete())
                return;
        }
        auto args = get_args();
        Tensor<T>& curr_data = this->data();
        Tensor<T>& curr_grad = this->grad();
        for (int i = 0; i < this->dependencies.size(); ++i) {
            auto dep = this->dependencies[i];
            if (!dep->requires_grad)
                continue;
            Tensor<T> local_grad(dep->grad().shape);
            source_functor_ptr->apply_backward(i, args, &curr_data, &curr_grad, &local_grad);
            dep->accumulate_grad(local_grad);
            if (recursive)
                dep->backward(this, true);
        }
    }

    template<typename T>
    vector<const Tensor<T>*> AutogradVariable<T>::get_args() const {
        vector<Tensor<T>> args;
        for (const auto &dep: this->dependencies)
            args.emplace_back(&dep->data());
        return args;
    }

    template<typename T>
    bool AutogradVariable<T>::is_root() const {
        return VariableBase<T>::is_root() && this->_data.size == 1;
    }

    template<typename T>
    AutogradVariable<T>::AutogradVariable(const string &name, const Functor<T> &source_functor, bool requires_grad) :
            VariableBase<T>(name, Tensor<T>(source_functor.output_shape), requires_grad),
            source_functor_ptr(source_functor.clone()) {}

}