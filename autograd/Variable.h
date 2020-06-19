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
    shared_ptr<Matrix> jac;
    vector<Variable*> dependencies;
    Vector data;

    void check_graph_integrity(unordered_set<Variable*>& visited);

public:
    explicit Variable(const Vector& data) : data(data){}
    Variable(const Vector& data, const Matrix& jac) :
        data(data), jac(jac.clone()) {}

    virtual ~Variable() {}

    Vector& get_data();

    void add_dependency(Variable* dep);
    virtual void accumulate_jac(const Matrix& jac) = 0;
    virtual void forward() = 0;
    virtual void backward(bool recursive=true) = 0;
    virtual void zero_jac(bool recursive=true) = 0;

    bool is_leaf();

    void check_graph_integrity();
};


#endif //TARGETPRACTICE_VARIABLE_H
