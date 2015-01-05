#include <cstring>
#include "inexact_newton.h"
#include "vec_op.h"
#include "log.h"

void InexactNewton::prepare_optimize(const Problem* data){
    n = data->n;
    numData = data->l;
    Real *mem = new Real[3 * n + numData + 2 * n + 3 * n];
    // self memory=3*n[w,g,d]
    w = mem;
    std::memset(w,0,sizeof(Real) * n); // init
    g = mem + n;
    d = mem + 2 * n;
    // memory=numData +1 [ predict prob, funcVal] 
    gradientCalc = new GradientCalc(data);
    gradientCalc->set_memory(mem + 3 * n);
    // memory=2*n[xp,gp]
    linearSearch = new LinearSearch(gradientCalc);
    linearSearch->set_memory(mem + 3 * n + numData, mem + 3 * n + numData + n);
    hessianMul = new HessianVecProduct(gradientCalc);
    if(sampleRatio > 0 && sampleRatio <= 1){
        hessianMul->set_sample_size((uint32_t)(sampleRatio * numData));
    }
    // memory=3*n[r,p,y]
    cgSolver = new CGSolver(hessianMul,n);
    cgSolver->set_memory(mem + 2 * n + numData + 2 * n,mem + 2 * n + numData + 3 * n, mem + 2 * n + numData + 4 * n);
}
void InexactNewton::post_optimize(){
    Optimizer::post_optimize();
    delete cgSolver;
}
void InexactNewton::optimize(){
    set_parameter(optimizePara);
    (*gradientCalc)(w,g);
    funcVal = gradientCalc->get_funcVal();
    gNorm = std::sqrt(vec_dot(g,g,n));
    numIter = 0;
    while(true){
        cgSolver->solve(d,cgPara,g,-1);
        hessianMul->next_sample();
        uint32_t cgIter = cgSolver->get_iter_num();
        double step_len = (*linearSearch)(w,g,d);
        if( step_len < 0 ){
            Log::error("InexactNewton::do_optimize","can't find suitable step size at line-search");
            break;
        }
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
void InexactNewton::unittest(){
    return;
    Log::raw("===========InexactNewton::unittest===============");
    Problem data = Problem::unittest();
    InexactNewton newton;
    newton.prepare_optimize(&data);
    std::memset(newton.w,0,sizeof(Real) * newton.n);
    newton.w[4] = 1;
    OptimizePara para;
    para.maxIter = 50;
    para.gNormKsi = 1e-12;
    newton.set_parameter(para);
    newton.optimize();
    newton.post_optimize();
    Log::raw("+++++++++++InexactNewton::unittest+++++++++++++++");

}
