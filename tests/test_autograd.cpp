//
// Created by LevZ on 6/17/2020.
//

#include "../distributions/distributions.h"
#include "../autograd/autograd.h"
#include "../autograd/Loss.h"

int main(){

    auto mu = randn(5, 10, 2);
    auto sigma = uniform(0, 1, 2);
    MultivariateGaussian gaussian(mu, sigma);
    auto gaussian_true = RandomVariable::make("true_dist", gaussian, false);
    auto [mu_pred, sigma_pred] = tuple {randn(5, 10, 2), uniform(0, 1, 2)};
    auto [mu_var, sigma_var] = tuple { Parameter::make("mu", mu_pred),
                                       Parameter::make("sigma", sigma_pred) };
    auto gaussian_pred = RandomVariable::make("pred_dist", MultivariateGaussian{mu_pred.shape()});
    gaussian_pred->add_dependency(mu_var);
    gaussian_pred->add_dependency(sigma_var);
    MSELoss criterion{mu_var->shape()};
    cout << "True distribution = " << gaussian << endl;
    auto loss = criterion(gaussian_true, gaussian_pred);
    for(int i=0; i < 10000000; ++i){
        loss->zero_grad(true);
        if (i % 100000 == 0)
            cout << "Epoch " << i+1 << ": loss= " << loss->forward_recursive().item() << "\t";
        loss->backward();
        mu_var->data() -= (7e-7 * mu_var->grad());
        sigma_var->data() -= (7e-7 * sigma_var->grad());
        if (i % 100000 == 0)
            cout << "mu_pred, sigma_pred = " << mu_var->data() << ", " << sigma_var->data() << endl;
    }
    cout << "final mu_pred, sigma_pred = " << mu_var->data() << ", " << sigma_var->data() << endl;
    cout << "real dist = " << gaussian << endl;

    return 0;
}