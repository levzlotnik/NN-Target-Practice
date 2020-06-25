//
// Created by LevZ on 6/16/2020.
//

#ifndef TARGETPRACTICE_VARIABLE_H
#define TARGETPRACTICE_VARIABLE_H

#include "../common.h"
#include <memory>
#include <utility>
#include "../BLAS/BLAS.h"

using namespace std;


class Variable {
protected:
    shared_ptr<Vector> ptr_grad_data;
    vector<Variable*> dependencies;
    vector<Variable*> dependees;
    Vector data;
    string name;

    void check_graph_integrity(unordered_set<Variable*>& visited);

    // Backpropagates the gradient to the current dependencies.
    // If recursive=true - backpropagates for all dependencies as well.
    virtual void backward(Variable *dependee, bool recursive) = 0;
    friend class AutogradVariable;
    friend class RandomVariable;

public:
    bool requires_grad;
    Variable(string name, Vector data, bool requires_grad = true)
            : name(std::move(name)), data(std::move(data)) , requires_grad(requires_grad){}
    Variable(string name, Vector data, const Vector& grad_data, bool requires_grad = true) :
            name(std::move(name)), data(std::move(data)),
            ptr_grad_data(grad_data.clone()), requires_grad(requires_grad) {}

    virtual ~Variable() = default;

    Vector& get_data();
    Vector& grad();

    void add_dependency(Variable* dep);
    virtual void accumulate_grad(const Vector &jac) = 0;
    virtual void forward() = 0;

    // Prepares the graph for backward operation. Should be called before
    // calling backward each time.
    virtual void prepare_backward() = 0;

    // Backpropagates through the entire graph and assigns gradients for every variable
    // according to the current gradient.
    void backward();

    virtual void zero_grad(bool recursive) = 0;

    // Returns true if this is a leaf of the graph - idx_proj.e. no dependencies.
    bool is_leaf() const;
    // Returns true if this is a root of the graph - idx_proj.e. no dependees.
    virtual bool is_root() const;

    void check_graph_integrity();
};


#endif //TARGETPRACTICE_VARIABLE_H
