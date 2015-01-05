#include <cstdlib>
#include <iostream>
#include "dataset.h"
#include "optimizer.h"
#include "inexact_newton.h"
#include "stochastic_gradient_descent.h"
#include "log.h"
#include "config.h"

int main(){
    config();
    Optimizer *optimizer = new StochGradientDescent();
    optimizer->set_parameter(para);
    optimizer->prepare_optimize(&data);
    optimizer->optimize();
    optimizer->post_optimize();
    delete optimizer;
    data.free_memory();
    return 0;
}

