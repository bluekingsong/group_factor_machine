#include <cstdlib>
#include <iostream>
#include "dataset.h"
#include "optimizer.h"
#include "inexact_newton.h"
#include "gradient_descent.h"
#include "lbfgs.h"
#include "log.h"
#include "config.h"

int main(){
    config();
    para.maxIter = 400;
    Optimizer *optimizer = new GradientDescent();
    optimizer->set_parameter(para);
    optimizer->prepare_optimize(&data);
    optimizer->optimize();
    optimizer->post_optimize();
    delete optimizer;
    data.free_memory();
    return 0;
}

