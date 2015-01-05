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
    Optimizer *optimizer = new LBFGS();
    int mems[] = {5,20,35,50} ;
    int n = 4;
    for(int i = 0; i < n; ++i){
        para.BFGS_m = mems[i];
        optimizer->set_parameter(para);
        optimizer->prepare_optimize(&data);
        optimizer->optimize();
        optimizer->post_optimize();
    }
    delete optimizer;
    data.free_memory();
    return 0;
}

