//
// Created by LevZ on 6/17/2020.
//

#include "../autograd/autograd.h"
using namespace blas;
using namespace autograd;

void test_autograd_simple(){
    cout << "TEST AUTOGRAD SIMPLE:" << endl;
    auto true_x = randn<double>(5, 10, 7);
    auto pred_x = zeros_like(true_x);
    auto pred_x_param = Parameter<double>::make("pred_x", pred_x, true);
    auto true_x_buffer = InputBuffer<double>::make("true_x", true_x);
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


int main(){
    test_autograd_simple();
//    test_autograd_ops();
//    test_vi_simple();
    return 0;
}