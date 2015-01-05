#include <cstdlib>
#include <iostream>
#include "dataset.h"
#include "optimizer.h"
#include "inexact_newton.h"
#include "gradient_descent.h"
#include "lbfgs.h"
#include "log.h"
#include "config.h"
#include "trust_region.h"

int main(){
    config();
    CGPara cgPara;
    cgPara.maxIter = 50, cgPara.zeroEps = 1e-8, cgPara.errorNormKsi = 0.1;
    cgPara.autoNormKsi = false, cgPara.checkPositiveDefined = true;
    cgPara.xNormKsi = -1;
    double sampleRatio = 1;
    Optimizer *optimizer = new TRON();
    //int iters[] = { 5,11,17,23};
    //int iters[] = { 29,35,41};
    //int iters[] = {47,53};
    int iters[] = {5,11,17,23,29,35,41,47,53};
    int n = 4;
    //double ratio[] = {1,0.05,0.1,0.2};
    double ratio[] = {0.0001,0.001,0.005,0.01,0.05,0.1};
    int m = 6;
//    for(int j = 0; j < m; ++j){
//    sampleRatio = ratio[j];
//    for(int i = 0; i < n; ++i){
//    cgPara.maxIter = iters[i];
    optimizer->set_parameter(para);
    optimizer->prepare_optimize(&data);
    ((TRON*)optimizer)-> set_para(cgPara,tronPara);
    optimizer->optimize();
    optimizer->post_optimize();
//    }
//    }
    delete optimizer;
    data.free_memory();
    return 0;
}

