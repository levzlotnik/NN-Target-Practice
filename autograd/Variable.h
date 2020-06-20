//
// Created by LevZ on 6/16/2020.
//

#ifndef TARGETPRACTICE_VARIABLE_H
#define TARGETPRACTICE_VARIABLE_H

#include "../common.h"
#include <memory>
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
    explicit Variable(const Vector& data, bool requires_grad = true) : data(data){}
    Variable(const Vector& data, const Vector& grad_data, bool requires_grad = true) :
            data(data), ptr_grad_data(grad_data.clone()) {}

    virtual ~Variable() {}

    Vector& get_data();
    Vector& grad();

    void add_dependency(Variable* dep);
    virtual void accumulate_grad(const Vector &jac) = 0;
    virtual void forward() = 0;
    virtual void backward(const Vector& grad, bool recursive=true) = 0;
    void backward();
    virtual void zero_grad(bool recursive=true) = 0;

    bool is_leaf();

    void check_graph_integrity();
};


#endif //TARGETPRACTICE_VARIABLE_H
