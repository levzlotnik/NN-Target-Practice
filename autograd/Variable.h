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
    vector<Variable*> dependencies;
    vector<Variable*> dependees;
    Vector _data;
    Vector _grad;
    string name;

    void check_graph_integrity(unordered_set<Variable*>& visited);

    // Backpropagates the gradient to the current dependencies.
    // If recursive=true - backpropagates for all dependencies as well.
    bool grad_accumulation_complete() const;
    virtual void backward(Variable *dependee, bool recursive) = 0;
    unordered_map<Variable*, int> unvisited_dependees;
    Vector forward_recursive();

    friend class AutogradVariable;
    friend class RandomVariable;

public:
    bool requires_grad;
    Variable(string name, Vector data, bool requires_grad = true)
            : name(std::move(name)), _data(std::move(data)),
              _grad(Vector::zeros_like(data)), requires_grad(requires_grad){}
    Variable(string name, Vector data, Vector  grad_data, bool requires_grad = true) :
            name(std::move(name)), _data(std::move(data)),
            _grad(std::move(grad_data)), requires_grad(requires_grad) {}

    virtual ~Variable() = default;

    Vector& data();
    Vector& grad();

    virtual void add_dependency(Variable* dep);
    void accumulate_grad(const Vector &grad);
    virtual Vector forward() = 0;

    // Prepares the graph for backward operation. Should be called before
    // calling backward each time.
    void prepare_backward();

    // Backpropagates through the entire graph and assigns gradients for every variable
    // according to the current gradient.
    void backward();

    void zero_grad(bool recursive);

    // Returns true if this is a leaf of the graph - idx_proj.e. no dependencies.
    bool is_leaf() const;
    // Returns true if this is a root of the graph - idx_proj.e. no dependees.
    virtual bool is_root() const;

    void check_graph_integrity();
};


#endif //TARGETPRACTICE_VARIABLE_H
