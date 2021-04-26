//
// Created by LevZ on 6/16/2020.
//

#include "VariableBase.h"
#include "Functor.h"
#include "VariableMath.h"

namespace autograd {

    template<typename T>
    void VariableBase<T>::add_dependency(const Variable<T>& dep) {
        if (dep.get() == nullptr) {
            warning::warn("Adding null dependency is ignored.");
            return;
        }
        dependencies.push_back(dep);
        dep->dependees.push_back(this);
    }


    template<typename T>
    inline Tensor<T>& VariableBase<T>::data() {
        return _data;
    }

    template<typename T>
    inline bool VariableBase<T>::is_leaf() const {
        return dependencies.empty();
    }

    template<typename T>
    void VariableBase<T>::check_graph_integrity() {
        unordered_set<VariableBase *> visited;
        check_graph_integrity(visited);
    }

    template<typename T>
    void VariableBase<T>::check_graph_integrity(unordered_set<VariableBase *>& visited) {
        if (visited.count(this) > 0)
            throw runtime_error("Graph contains cycles. Redefine the graph to not contain cycles.");
        visited.insert(this);
        for (const auto& dep: dependencies)
            dep->check_graph_integrity(visited);
        visited.erase(this);
    }

    template<typename T>
    inline Tensor<T>& VariableBase<T>::grad() {
        return _grad;
    }

    template<typename T>
    bool VariableBase<T>::is_root() const {
        return dependees.empty();
    }

    template<typename T>
    void VariableBase<T>::backward() {
        if (!is_root())
            warning::warn("Calling '.backward()' on a non-root variable will force the gradients to"
                          "flow from the middle of the graph, and may produce unexpected results. "
                          "Avoid it unless you really know what you're doing.");
        prepare_backward();
        grad().fill_(T(1));
        backward(nullptr, true);
    }

    template<typename T>
    void VariableBase<T>::accumulate_grad(const Tensor<T>& grad) {
        if (requires_grad)
            _grad += grad;
    }

    template<typename T>
    void VariableBase<T>::prepare_backward() {
        unvisited_dependees.clear();
        for (auto dependeePtr: dependees)
            unvisited_dependees[dependeePtr]++;
        for (const auto& dep: dependencies)
            dep->prepare_backward();
    }


    template<typename T>
    inline bool VariableBase<T>::grad_accumulation_complete() const {
        for (auto[dep_ptr, required_visits] : unvisited_dependees)
            if (required_visits > 0)
                return false;
        return true;
    }


    template<typename T>
    void VariableBase<T>::zero_grad(bool recursive) {
        if (!requires_grad)
            return;
        _grad.fill_(T(0));
        if (recursive)
            for (const auto& dep: dependencies)
                dep->zero_grad(true);
    }


    template<typename T>
    Tensor<T>&  VariableBase<T>::forward_recursive() {
        for (const auto& dep: dependencies)
            dep->forward_recursive();
        return forward();
    }

    template<typename T>
    GraphvizPrinter& VariableBase<T>::gather_connection_graphviz(GraphvizPrinter& gvzp) {
        gvzp.create_node(this->name, this->node_style_graphviz());
        for (const auto& dep: dependencies) {
            dep->gather_connection_graphviz(gvzp);
            gvzp.create_dependency(this->name, dep->name);
        }
        return gvzp;
    }

    template<typename T>
    ostream& VariableBase<T>::print_graphviz(ostream& os) {
        GraphvizPrinter gvzp;
        gather_connection_graphviz(gvzp);
        return gvzp.print_dot(os);
    }

    template<typename T>
    string VariableBase<T>::node_style_graphviz() {
        return "";
    }

    template<typename T>
    void VariableBase<T>::remove_dependency(const Variable<T>& dep) {
        vector<VariableBase *>& dep_dependees = dep->dependees;
        auto it_dep = std::remove(dep_dependees.begin(), dep_dependees.end(), this);
        dep_dependees.erase(it_dep, dep_dependees.end());
        auto it_this = std::remove_if(dependencies.begin(), dependencies.end(),
                                      [dep](const Variable<T>& v) { return v.equals(dep); });
        dependencies.erase(it_this, dependencies.end());
    }

    template<typename T>
    VariableBase<T>::~VariableBase() {
        vector<Variable<T>> dependencies_clone(dependencies);
        for (const Variable<T>& dep: dependencies_clone)
            this->remove_dependency(dep);
        for (VariableBase *dependee: dependees) {
            auto it_dependee = std::remove_if(dependee->dependencies.begin(),
                                              dependee->dependencies.end(),
                                              [this](const Variable<T>& v) { return v.get() == this; });
            dependee->dependencies.erase(it_dependee, dependee->dependencies.end());
        }
    }

#define DEF_VARIABLE_MATH_METHOD(func) \
    template<typename T> Variable<T> Variable<T>::func() const { return autograd::func(*this); }

    MACRO_MATH_FUNCTIONS(DEF_VARIABLE_MATH_METHOD)

#define INSTANTIATE_TEMPLATE_VARIABLE(dtype) \
    template class VariableBase<dtype>;      \
    template class Variable<dtype>;

    INSTANTIATE_TEMPLATE_VARIABLE(double)
    INSTANTIATE_TEMPLATE_VARIABLE(float)

}