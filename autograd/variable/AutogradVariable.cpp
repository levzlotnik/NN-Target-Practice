//
// Created by LevZ on 6/17/2020.
//

#include "AutogradVariable.h"


Vector AutogradVariable::forward() {
    if (!is_leaf())
         _data = (*this->source_functor_ptr)(get_args());
    return _data;
}

void AutogradVariable::backward(VariableBase *dependee, bool recursive) {
//    cout << name << ".backward()" << endl;
    if (!requires_grad)
        return;
    if (dependee)
    {
        if (unvisited_dependees.at(dependee) <= 0)
            throw runtime_error("WTF... Did you forget to call '.prepare_backward()'? "
                                "Maybe '.check_graph_integrity()'?");

        unvisited_dependees[dependee]--;
        if (!grad_accumulation_complete())
            return;
    }
    auto args = get_args();
    for(int i=0; i < dependencies.size(); ++i) {
        auto dep = dependencies[i];
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

vector<Vector> AutogradVariable::get_args() {
    vector<Vector> args;
    for (const auto& dep: dependencies)
        args.emplace_back(dep->data());
    return args;
}

bool AutogradVariable::is_root() const {
    return VariableBase::is_root() && _data.n == 1;
}

AutogradVariable::AutogradVariable(string name, const Functor &source_functor, bool requires_grad) :
        VariableBase(std::move(name), Vector(source_functor.output_shape), requires_grad),
        source_functor_ptr(source_functor.clone()) {}

