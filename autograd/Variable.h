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
    Vector data;

    void check_graph_integrity(unordered_set<Variable*>& visited);

public:
    bool requires_grad;
    explicit Variable(Vector data, bool requires_grad = true) : data(std::move(data)) ,requires_grad(requires_grad){}
    Variable(Vector data, const Vector& grad_data, bool requires_grad = true) :
            data(std::move(data)), ptr_grad_data(grad_data.clone()), requires_grad(requires_grad) {}

    virtual ~Variable() = default;

    Vector& get_data();
    Vector& grad();

    void add_dependency(Variable* dep);
    virtual void accumulate_grad(const Vector &jac) = 0;
    virtual void forward() = 0;
    virtual void backward(const Vector& current_grad, bool recursive=true) = 0;
    virtual void zero_grad(bool recursive=true) = 0;

    bool is_leaf();

    void check_graph_integrity();
};


#endif //TARGETPRACTICE_VARIABLE_H
