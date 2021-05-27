//
// Created by LevZ on 6/17/2020.
//

#include "../autograd/autograd.h"
using namespace blas;
using namespace autograd;

void test_autograd_simple()
{
    cout << "TEST AUTOGRAD SIMPLE:" << endl;
    auto true_x = randn<double>(5, 10, {7});
    auto pred_x = zeros_like(true_x);
    auto pred_x_param = Parameter<double>::make("pred_x", pred_x);
    auto true_x_buffer = InputBuffer<double>::make("true_x", true_x);
    MSELoss<double> criterion{pred_x.shape};
    auto loss = criterion(true_x_buffer, pred_x_param);
    GraphvizPrinter gvzp;
    loss->gather_connection_graphviz(gvzp);
    gvzp.export_to("simple.svg");
    for (int i = 0; i < 1000; ++i)
    {
        auto loss_val = loss->forward_recursive().item();
        loss->zero_grad(true);
        if (i % 100 == 0)
            cout << "Epoch " << i + 1 << ": loss= " << loss_val << "\t";
        loss->backward();
        pred_x_param->data() -= (1e-1 * pred_x_param->grad());
        if (i % 100 == 0)
            cout << " pred_x_param = " << pred_x_param->data() << endl;
    }
    cout << "true_x_buffer, pred_x_param = " << true_x_buffer->data() << ", " << pred_x_param->data() << endl;
}

void test_autograd_linear_regression()
{
    cout << "TEST AUTOGRAD LINEAR REGRESSION:" << endl;
    double alpha = 1e-3;
    auto true_theta = arange<double>(1, 4);
    auto pred_theta = zeros_like(true_theta);
    auto pred_theta_param = Parameter<double>::make("pred_theta", pred_theta);
    auto true_theta_param = Constant<double>::make("true_theta", true_theta);
    auto input = InputBuffer<double>::make("input", arange<double>(0, 5 * 3).const_view({5, 3}));
    auto true_y = true_theta_param * input;
    auto pred_y = pred_theta_param * input;
    cout << "true_theta, pred_theta = " << true_theta_param.data() << ", " << pred_theta_param.data() << endl;
    MSELoss<double> criterion(pred_y.shape());
    auto loss = criterion(pred_y, true_y);
    GraphvizPrinter gvzp;
    loss->gather_connection_graphviz(gvzp);
    gvzp.export_to("linregress.svg");
    for (int i = 0; i < 1000; ++i)
    {
        loss->forward_recursive();
        auto loss_val = loss->data().item();
        loss->zero_grad(true);
        loss->backward();
        pred_theta_param.data() -= (alpha * pred_theta_param.grad());
        if (i % 100 == 0)
        {
            cout << "Epoch " << i + 1 << ": loss= " << loss_val << "\t";
            cout << " pred_theta_param = " << pred_theta_param.data() << "\t";
#ifndef NDEBUG
            cout << " pred_theta_param.grad = " << pred_theta_param.grad() << endl;
#endif
        }
    }

    cout << "true_theta, pred_theta = " << true_theta << ", " << pred_theta_param.data() << endl;
}

void test_autograd_manual_linear_regression()
{
    cout << "TEST AUTOGRAD MANUAL LINEAR REGRESSION:" << endl;
    double alpha = 1e-1;
    auto true_theta = arange<double>(1, 4);
    auto pred_theta = zeros_like(true_theta);
    auto pred_theta_param = Parameter<double>::make("pred_theta", pred_theta);
    auto true_theta_param = Constant<double>::make("true_theta", true_theta);
    auto input = InputBuffer<double>::make("input", arange<double>(0, 5 * 3).const_view({5, 3}));
    auto true_y = true_theta_param * input;
    auto pred_y = pred_theta_param * input;
    cout << "true_theta, pred_theta = " << true_theta_param.data() << ", " << pred_theta_param.data() << endl;
    auto dy = pred_y - true_y;
    auto dy2 = pow(dy, 2.0);
    auto loss = sum(dy2) / (double)dy2->data().size;
    GraphvizPrinter gvzp;
    loss->gather_connection_graphviz(gvzp);
    gvzp.export_to("manual_linregress.svg");
    for (int i = 0; i < 1000; ++i)
    {
        loss->forward_recursive();
        auto loss_val = loss->data().item();
        loss->zero_grad(true);
        loss->backward();
        pred_theta_param.data() -= (alpha * pred_theta_param.grad());
        if (i % 100 == 0)
        {
            cout << "Epoch " << i + 1 << ": loss= " << loss_val << "\t";
            cout << " pred_theta_param = " << pred_theta_param.data() << "\t";
#ifndef NDEBUG
            cout << " pred_theta_param.grad = " << pred_theta_param.grad() << endl;
#endif
        }
    }

    cout << "true_theta, pred_theta = " << true_theta << ", " << pred_theta_param.data() << endl;
}


void test_multi_layer_perceptron()
{
    cout << "TEST AUTOGRAD MLP:" << endl;
    double alpha = 5e-2;
    auto x = linspace<double>(-1, 1, 500).const_view({500, 1});
    auto y = x*x;
    auto input = InputBuffer<double>::make("x", x.const_view({-1, 1}));
    auto w1 = Parameter<double>::make("w1", uniform(-1., 1., {1, 8})),
         b1 = Parameter<double>::make("b1", ones<double>({8}));
    auto w2 = Parameter<double>::make("w2", uniform(-1., 1., {8, 8})),
         b2 = Parameter<double>::make("b2", ones<double>({8}));
    auto w3 = Parameter<double>::make("w3", uniform(-1., 1., {8, 8})),
         b3 = Parameter<double>::make("b3", ones<double>({8}));
    auto w4 = Parameter<double>::make("w4", uniform(-1., 1., {8, 8})),
         b4 = Parameter<double>::make("b4", ones<double>({8}));
    auto w5 = Parameter<double>::make("w5", uniform(-1., 1., {8, 1})),
         b5 = Parameter<double>::make("b5", ones<double>({1}));
    vector<Variable<double>> params = {w1, b1, w2, b2, w3, b3, w4, b4, w5, b5};
    auto h1 = relu(matmul(input, w1) + b1);
    auto h2 = relu(matmul(h1, w2) + b2);
    auto h3 = relu(matmul(h2, w3) + b3);
    auto h4 = relu(matmul(h3, w4) + b4);
    auto y_pred = matmul(h4, w5) + b5;
    MSELoss<double> criterion{y.shape};
    GraphvizPrinter gvzp;
    auto y_true = InputBuffer<double>::make("y_true", y);
    auto loss = criterion(y_pred, y_true); 
    loss->gather_connection_graphviz(gvzp);
    gvzp.export_to("MLP.svg");
    for (int i = 0; i < 100; ++i)
    {
        loss->forward_recursive();
        auto loss_val = loss->data().item();
        loss->zero_grad(true);
        loss->backward();
        for (const auto& p: params) {
            p.data() -= alpha * p.grad();
        }
        if (i % 10 == 0) 
        {
            cout << "Epoch " << i << " loss = " << loss_val << endl;
            alpha *= 0.8;
        }
    }
}

int main()
{
    test_autograd_simple();
    test_autograd_linear_regression();
    test_autograd_manual_linear_regression();
    test_multi_layer_perceptron();
    return 0;
}