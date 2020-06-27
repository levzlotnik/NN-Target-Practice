//
// Created by LevZ on 6/16/2020.
//

#ifndef TARGETPRACTICE_VARIABLEBASE_H
#define TARGETPRACTICE_VARIABLEBASE_H

#include "../../common.h"
#include <memory>
#include <utility>
#include "../../BLAS/BLAS.h"

using namespace std;

class VariableBase;

class Variable : public shared_ptr<VariableBase> {
public:
    using shared_ptr<VariableBase>::shared_ptr;
};

class VariableBase {
protected:
    vector<shared_ptr<VariableBase>> dependencies;
    vector<VariableBase*> dependees;
    Vector _data;
    Vector _grad;
    string name;

    void check_graph_integrity(unordered_set<VariableBase*>& visited);

    // Backpropagates the gradient to the current dependencies.
    // If recursive=true - backpropagates for all dependencies as well.
    bool grad_accumulation_complete() const;
    virtual void backward(VariableBase *dependee, bool recursive) = 0;
    unordered_map<VariableBase*, int> unvisited_dependees;

    friend class AutogradVariable;
    friend class RandomVariable;

    VariableBase(string name, Vector data, Vector  grad_data, bool requires_grad = true) :
            name(std::move(name)), _data(std::move(data)),
            _grad(std::move(grad_data)), requires_grad(requires_grad) {}

    VariableBase(string name, const Vector& data, bool requires_grad = true)
            : VariableBase(std::move(name), data, Vector::zeros_like(data), requires_grad){}
public:
    bool requires_grad;

    virtual ~VariableBase() = default;

    Vector& data();
    Vector& grad();
    inline int shape() const { return _data.shape(); }

    virtual void add_dependency(const Variable &dep);
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

    virtual bool is_param() const { return false; }
    virtual bool is_input_buffer() const { return false; }

    void check_graph_integrity();

    Vector forward_recursive();
};



#endif //TARGETPRACTICE_VARIABLEBASE_H
