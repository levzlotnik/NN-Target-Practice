//
// Created by LevZ on 6/17/2020.
//

#include "../autograd/autograd.h"
using namespace blas;
using namespace autograd;

void test_autograd_simple(){
    cout << "TEST AUTOGRAD SIMPLE:" << endl;
    auto true_x = randn<double>(5, 10, {7});
    auto pred_x = zeros_like(true_x);
    auto pred_x_param = Parameter<double>::make("pred_x", pred_x, true);
    auto true_x_buffer = InputBuffer<double>::make("true_x", true_x);
    MSELoss<double> criterion{pred_x.shape};
    auto loss = criterion(true_x_buffer, pred_x_param);
    GraphvizPrinter gvzp;
    loss->gather_connection_graphviz(gvzp);
    gvzp.export_to("svg");
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

void test_autograd_linear_regression() {
    cout << "TEST AUTOGRAD LINEAR REGRESSION:" << endl;
    auto true_theta = arange<double>(1, 4);
    auto pred_theta = zeros_like(true_theta);
    auto pred_theta_param = Parameter<double>::make("pred_theta", pred_theta);
    auto true_theta_param = Constant<double>::make("true_theta", true_theta);
    auto input = InputBuffer<double>::make("input", randn<double>({5, 3}));
    auto true_y = true_theta_param * input;
    auto pred_y = pred_theta_param * input;
    cout << "true_theta, pred_theta = " << true_theta_param.data() << ", " << pred_theta_param.data() << endl;
    MSELoss<double> criterion{pred_y.shape()};
    auto loss = criterion(pred_y, true_y);
    GraphvizPrinter gvzp;
    loss->gather_connection_graphviz(gvzp);
    gvzp.export_to("svg");
    for(int i=0; i < 1000; ++i){
        loss->zero_grad(true);
        loss->backward();
        pred_theta_param.data() -= (1e-1 * pred_theta_param.grad());
        if (i % 100 == 0) {
            cout << "Epoch " << i + 1 << ": loss= " << loss->forward_recursive().item() << "\t";
            cout << " pred_theta_param = " << pred_theta_param.data() << "\t";
            cout << " pred_theta_param.grad = " << pred_theta_param.grad() << endl;
        }
    }

    cout << "true_theta, pred_theta = " <<
         true_theta << ", " <<
         pred_theta_param.data() << endl;
}

int main(){
    test_autograd_simple();
    test_autograd_linear_regression();
    return 0;
}