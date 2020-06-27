//
// Created by LevZ on 6/17/2020.
//

#include "../distributions/distributions.h"
#include "../autograd/autograd.h"
#include "../autograd/Loss.h"

void test_autograd_simple(){
    cout << "TEST AUTOGRAD SIMPLE:" << endl;
    auto true_x = randn(5, 10, 7);
    auto pred_x = Vector::zeros_like(true_x);
    auto pred_x_param = Parameter::make("pred_x", pred_x, true);
    auto true_x_buffer = InputBuffer::make("true_x", true_x);
    MSELoss criterion{pred_x.shape()};
    auto loss = criterion(true_x_buffer, pred_x_param);
    for(int i=0; i < 1000; ++i){
        loss->zero_grad(true);
        if (i % 100 == 0)
            cout << "Epoch " << i+1 << ": loss= " << loss->forward_recursive().item() << "\t";
        loss->backward();
        pred_x_param->data() -= (1e-1 * pred_x_param->grad());
        if (i % 100 == 0)
            cout << " pred_x_param = " << pred_x_param->data() << endl;
    }
    cout << "true_x_buffer, pred_x_param = " <<
        true_x_buffer->data() << ", " <<
        pred_x_param->data() << endl;
}

void test_autograd_ops(){
    cout << "TEST AUTOGRAD OPS" << endl;
    auto x = Vector::linspace(-1, 1, 500);
    auto y = x * 3 + 5;
    auto [x_buf, y_true] = tuple {InputBuffer::make("x_buf", x), InputBuffer::make("y_true", y)};
    auto [a, b] = tuple
        { Parameter::make("a", randn(0, 1, 1)),
          Parameter::make("b", randn(0, 1, 1))};
    auto y_pred = x_buf * a + b;
    auto loss = mean(pow(y_pred - y_true, 2)); // The problem is here!
//    auto loss = mse(y_true, y_pred); // This works!
    float alpha = 1e-1f;
    for(int i=0; i < 100; ++i){
        loss->zero_grad(true);
        if (i % 10 == 0)
            cout << "Epoch " << i+1 << ": loss= " << loss->forward_recursive().item() << "\t";
        loss->backward();
        a->data() -= (alpha*a->grad());
        b->data() -= (alpha*b->grad());
        if (i % 10 == 0) {
            cout << " a.data, b.data = " << a->data() << ", " << b->data() << endl;
            cout << " a.grad, b.grad = " << a->grad() << ", " << b->grad() << endl;
        }
    }
}

void test_vi_simple(){
    cout << "TEST VI SIMPLE" << endl;
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
        if (i % 1000000 == 0)
            cout << "Epoch " << i+1 << ": loss= " << loss->forward_recursive().item() << "\t";
        loss->backward();
        mu_var->data() -= (7e-7 * mu_var->grad());
        sigma_var->data() -= (7e-7 * sigma_var->grad());
        if (i % 1000000 == 0)
            cout << "mu_pred, sigma_pred = " << mu_var->data() << ", " << sigma_var->data() << endl;
    }
    cout << "final mu_pred, sigma_pred = " << mu_var->data() << ", " << sigma_var->data() << endl;
    cout << "real dist = " << gaussian << endl;
}

int main(){
    test_autograd_simple();
    test_autograd_ops();
    test_vi_simple();
    return 0;
}