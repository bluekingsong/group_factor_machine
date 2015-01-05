#include <cmath>
#include <cstring>
#include "vec_op.h"
#include "stochastic_gradient_descent.h"
#include "log.h"

void StochGradientDescent::optimize(){
    uint32_t batchSize = 50;
    double eta = 5;
    set_parameter(optimizePara);
    gradientCalc->set_sample_enabled(true);
    gradientCalc->set_sample_para(batchSize);
    funcVal = (*gradientCalc)(w,g);
    gNorm = std::sqrt(vec_dot(g,g,n));
    uint32_t total_num_iter = numData / batchSize;
    for(uint32_t numIter = 0; numIter < total_num_iter; ++numIter){
        vec_add(w,w,g,n,1,-eta / (numIter + 1));
        gradientCalc->next_sample();
        if(numIter % 1000 == 0){
            gradientCalc->set_sample_enabled(false);
            funcVal = (*gradientCalc)(w,g);
            Log::raw(make_monitor_str());
        }
        gradientCalc->set_sample_enabled(true);
        funcVal = (*gradientCalc)(w,g);
    }
}
void StochGradientDescent::prepare_optimize(const Problem* data){
    n = data->n;
    numData = data->l;
    Real *mem = new Real[2 * n + numData];
    // self memory=3*n[w,g,d] + l
    w = mem;
    std::memset(w,0,sizeof(Real) * n); // init
    g = mem + n;
    d = 0;
    // memory=numData +1 [ predict prob, funcVal]
    gradientCalc = new GradientCalc(data);
    gradientCalc->set_memory(mem + 2 * n);
}
