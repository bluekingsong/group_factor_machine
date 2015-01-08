#include <cmath>
#include <cstring>
#include "vec_op.h"
#include "gradient_descent.h"
#include "log.h"

void GradientDescent::optimize(){
    Optimizer::optimize();
    while(true){
        double initStep = guessInitStep(g,n,numIter);
        linearSearch->set_init_step(initStep);
        double step_len = (*linearSearch)(w,g,g,-1);
        if( step_len < 0 ){
            Log::error("InexactNewton::do_optimize","can't find suitable step size at line-search");
            break;
        }
        uint32_t numSearch = linearSearch->get_num_search();
        numLS += numSearch;
        numAccessData += numSearch * numData;
        vec_cpy(w,linearSearch->get_new_parameter(),n);
        vec_cpy(g,linearSearch->get_new_gradient(),n);
        if(check_stop_condition(linearSearch->get_new_funcVal())){
            break;
        }
        funcVal = linearSearch->get_new_funcVal();
        gNorm = std::sqrt(vec_dot(g,g,n));
        Log::raw(make_monitor_str());
    }
}
double GradientDescent::guessInitStep(const Real *g,uint32_t n,uint32_t numIter){
    double result = std::abs(g[0]);
    for(uint32_t i = 1; i < n; ++i){
        if(std::abs(g[i]) > result)    result = std::abs(g[i]);
    }
    return 1.0 / result * std::pow(0.9,(double)numIter);
}
void GradientDescent::prepare_optimize(GradientCalc* _gradientCalc,const Real* initW){
    gradientCalc = _gradientCalc;
    n = gradientCalc->get_parameter_size();
    numData = gradientCalc->get_instance_num();
    Real *mem = new Real[2 * n + numData + 2 * n];
    // self memory=3*n[w,g,d] + l
    w = mem;
    if(0 == initW){
        std::memset(w,0,sizeof(Real) * n); // init
    }else{
        vec_cpy(w,initW,n,1);
    }
    g = mem + n;
    d = 0;
    // memory=numData +1 [ predict prob, funcVal]
    gradientCalc->set_memory(mem + 2 * n);
    // memory=2*n[xp,gp]
    linearSearch = new LinearSearch(gradientCalc);
    linearSearch->set_memory(mem + 2 * n + numData, mem + 2 * n + numData + n);
}

std::string GradientDescent::report_algo_para()const{
    std::stringstream ss;
    ss<<"algo="<<optAlgoName<<","<<optimizePara.report()<<",memory="<<(sizeof(Real)*(4*n+numData)>>30)<<"G";
    return ss.str();
}
