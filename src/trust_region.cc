#include <cstring>
#include "inexact_newton.h"
#include "vec_op.h"
#include "log.h"
#include "trust_region.h"

void TRON::optimize(){
    set_parameter(optimizePara);
    (*gradientCalc)(w,g);
    funcVal = gradientCalc->get_funcVal();
    gNorm = std::sqrt(vec_dot(g,g,n));
    double delta = std::sqrt(gNorm);
    numIter = 0;
    while(true){
        CGPara tPara(cgPara);
        tPara.xNormKsi = delta;
        cgSolver->solve(d,tPara,g,-1);
        hessianMul->next_sample();
        uint32_t cgIter = cgSolver->get_iter_num();
        double step_len = (*linearSearch)(w,g,d);
        if( step_len < 0 ){
            Log::error("InexactNewton::do_optimize","can't find suitable step size at line-search");
            break;
        }
        delta = update_trust_region_radius(delta);
        uint32_t numSearch = linearSearch->get_num_search();
        numLS += numSearch;
        numAccessData += (numSearch + cgIter) * numData;
        vec_cpy(w,linearSearch->get_new_parameter(),n);
        vec_cpy(g,linearSearch->get_new_gradient(),n);
        if(check_stop_condition(linearSearch->get_new_funcVal())){
            break;
        }
        funcVal = linearSearch->get_new_funcVal();
        gNorm = std::sqrt(vec_dot(g,g,n));
        Log::raw(make_monitor_str());
    }
};
double TRON::update_trust_region_radius(double oldDelta)const{
    double rho = (linearSearch->get_new_funcVal() - funcVal);
    Real * t = w;
    (*hessianMul)(d,n,t);
    double modelDec = vec_dot(g,d,n) + 0.5 * vec_dot(d,t,n);
    rho /= modelDec;
    double low,high;
    if(rho <= tronPara.yita1){
        double t = std::sqrt(vec_dot(d,d,n));
        if(t > oldDelta)    t = oldDelta;
        low = tronPara.sigma1 * t;
        high = tronPara.sigma2 * oldDelta;
    }else if( rho < tronPara.yita2){
        low = tronPara.sigma1 * oldDelta;
        high = tronPara.sigma3 * oldDelta;
    }else{
        low = oldDelta;
        high = tronPara.sigma3 * oldDelta;
    }
    return (low + high) / 2;
}
