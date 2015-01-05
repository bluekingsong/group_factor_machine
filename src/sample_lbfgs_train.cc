#include <cstdlib>
#include <iostream>
#include "dataset.h"
#include "optimizer.h"
#include "inexact_newton.h"
#include "gradient_descent.h"
#include "sample_lbfgs.h"
#include "log.h"
#include "config.h"

int main(){
    config();
    Optimizer *optimizer = new SampleLBFGS();
    CGPara cgPara;
    cgPara.maxIter = 10, cgPara.zeroEps = 1e-8, cgPara.errorNormKsi = 0.1;
    cgPara.autoNormKsi = true, cgPara.checkPositiveDefined = true;
    cgPara.xNormKsi = -1; // no effect
    //double ratio[] = {0.005,0.01,0.05,0.1};
    double ratio[] = {0.0001,0.001};
    double sampleRatio = 0.01;
    int mems[] = {5,20,35} ;
    int n = 2;
    for(int j = 0; j < 3; ++j){
        para.BFGS_m = mems[j];
    for(int i = 0; i < n; ++i){
        sampleRatio = ratio[i];
        optimizer->set_parameter(para);
        optimizer->prepare_optimize(&data);
        ((SampleLBFGS*)optimizer)->set_sample_parameter(cgPara,sampleRatio);
        optimizer->optimize();
        optimizer->post_optimize();
    }
    }
    delete optimizer;
    data.free_memory();
    return 0;
}

