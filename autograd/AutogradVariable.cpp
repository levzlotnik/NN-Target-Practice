//
// Created by LevZ on 6/17/2020.
//

#include "AutogradVariable.h"
namespace autograd {
    template<typename T>
    Tensor<T> AutogradVariable<T>::forward() {
        if (!this->is_leaf())
            this->_data = (*this->source_functor_ptr)(get_args());
        return this->_data;
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
        for (int i = 0; i < this->dependencies.size(); ++i) {
            auto dep = this->dependencies[i];
//        cout << "dep[" << i << "] = " << dep->name << endl;
            if (!dep->requires_grad)
                continue;
            auto jac = source_functor_ptr->jac(i, args, this->_data);
//        cout << name << ".jac(" << i << ") = " << jac << endl;
            auto dep_grad = matmul(this->grad(), jac);
//        cout << name << ".dep[" << i << "]_grad = " << dep_grad << endl;
            dep->accumulate_grad(dep_grad);
            if (recursive)
                dep->backward(this, true);
        }
    }

    template<typename T>
    vector<Tensor<T>> AutogradVariable<T>::get_args() {
        vector<Tensor<T>> args;
        for (const auto &dep: this->dependencies)
            args.emplace_back(dep->data());
        return args;
    }

    template<typename T>
    bool AutogradVariable<T>::is_root() const {
        return VariableBase<T>::is_root() && this->_data.size == 1;
    }

    template<typename T>
    AutogradVariable<T>::AutogradVariable(const string &name, const Functor &source_functor, bool requires_grad) :
            VariableBase<T>(name, Tensor<T>(source_functor.output_shape), requires_grad),
            source_functor_ptr(source_functor.clone()) {}

}