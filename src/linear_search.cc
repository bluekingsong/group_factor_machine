#include <cmath>
#include <assert.h>
#include "linear_search.h"
#include "vec_op.h"
#include "log.h"

double LinearSearch::operator ()(const Real *x,const Real *g, const Real *d, double dFactor){
    double dec = dFactor * vec_dot(g,d,n);
    if(dec >= 0){ // non suitable step,p is not a descent search direction
        Log::warn("linearSearch","dFactor*d is not a descent search direction");
        return -1;
    }
    double alpha = get_init_step();
    vec_add(xp,x,d,n,1,alpha * dFactor);
    double old_funVal = gradientCalc->get_funcVal();
    funcVal = (*gradientCalc)(xp,gp);
    numIter = 1;
    while( funcVal > old_funVal + alpha * c1 * dec ){
        alpha *= scaleRatio;
        vec_add(xp,x,d,n,1,alpha * dFactor);
        funcVal = (*gradientCalc)(xp,gp);
        ++numIter;
        if(numIter % 10 == 0){
            std::cerr<<"[WRN]:LinearSearch,so much linear search iter="<<numIter<<std::endl;
        }
    }
    return alpha;
}
void LinearSearch::unittest(GradientCalc * gradientCalc){
    Log::raw("===========LinearSearch::unittest===============");
    LinearSearch *linearSearch = new LinearSearch(gradientCalc);
    uint32_t n = gradientCalc->get_parameter_size();
    Real *mem = new Real[4 * n];
    linearSearch->set_memory(mem,mem + n);
    Real *w = mem + 2 * n;
    Real *g = mem + 3 * n;
    w[0] = w[1] = w[2] = w[3] = 0;
    w[4] = 1;
    double f = (*gradientCalc)(w,g);
    double alpha = (*linearSearch)(w,g,g,-1);
    double fNew = linearSearch->get_new_funcVal();
    double xp_target[] = { -1.9621,    -0.2311,    -0.5000,    -2.2311,    -1.9242};
    std::cout<<"new parameter:";
    for(int i = 0; i < 5; ++i){
        std::cout<<linearSearch->get_new_parameter()[i]<<" ";
        assert(std::abs(xp_target[i] - linearSearch->get_new_parameter()[i]) < 1e-4);
    }
    std::cout<<std::endl;
    std::cout<<"f="<<f<<" fNew="<<fNew<<" alpha="<<alpha<<std::endl;
    assert(std::abs(fNew - 3.83054) < 1e-5);
    assert(std::abs(alpha - 1) < 1e-12);
    delete linearSearch;
    delete []mem;
    Log::raw("+++++++++++LinearSearch::unittest+++++++++++++++");
}
