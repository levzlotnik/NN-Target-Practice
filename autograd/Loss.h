//
// Created by LevZ on 6/26/2020.
//

#ifndef TARGETPRACTICE_LOSS_H
#define TARGETPRACTICE_LOSS_H


#include "variable/Variable.h"
#include "Functor.h"

class MSELoss : public Functor {
public:
    explicit MSELoss(int shape) : Functor({shape, shape}, 1, "MSELoss") {}
    Vector operator()(const vector<Vector> &args) const override;
    Vector operator()(const Vector& val_true, const Vector& val_pred) const;
    shared_ptr<Variable> operator()(const shared_ptr<Variable>& var_true, const shared_ptr<Variable>& var_pred,
            bool requires_grad = true) const;

    Matrix jac(int i, const vector<Vector> &inputs, const Vector &output) const override;

    Functor *clone() const override;
};


#endif //TARGETPRACTICE_LOSS_H
