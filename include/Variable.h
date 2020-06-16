//
// Created by LevZ on 6/16/2020.
//

#ifndef BLAS_VARIABLE_H
#define BLAS_VARIABLE_H

#include "../common.h"
#include <memory>
#include "Vector.h"
#include "Matrix.h"
#include "VectorFunction.h"

using namespace std;

class Variable {
protected:
    unique_ptr<Matrix> ptrJac;
    vector<Variable*> dependencies;
    unique_ptr<VectorFunction> functor;
    Vector data;

public:
    explicit Variable(Vector data) : data(std::move(data)), ptrJac(nullptr) {}
    Variable(Vector data, Matrix jac) : data(std::move(data)), ptrJac(make_unique<Matrix>(std::move(jac))) {}

    Vector& get_data();

    void add_dependency(Variable* dep);
    void accumulate_jac(const Matrix& jac);
    void forward();
    void backward(bool recursive=true);
    void zero_grad();

    bool is_leaf();
};


#endif //BLAS_VARIABLE_H
