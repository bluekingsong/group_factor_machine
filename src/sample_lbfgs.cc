#include <cstring>
#include <iostream>
#include "linear_search.h"
#include "gradient_descent.h"
#include "vec_op.h"
#include "sample_lbfgs.h"
#include "log.h"

// L-BFGS two-loop recursion
// @s the difference of Xk+1 and Xk for k=0 to m-1
// @y the difference of Gk+1 and Gk for k=0 to m-1
// @rho 1.0/(Yk'*Sk)
// @alpha alpha variable in the two-loop
// @g gradient at current iteration
// @n the number of parameter
// @m we keep at most latest m iteration information
// @gamma the initail approximation of inverse Hessian using gamma*I
// @result the bfgs direciton we need
void SampleLBFGS::calc_bfgs_direction(uint32_t k,double gamma){
    double *q = d;
    vec_cpy(q,g,n);
    uint32_t beg = (k-1) % m, end = 0;
    if( k > m)    end = k % m;
    for(uint32_t i = beg; true ; i = (m+i-1) % m){
        alpha[i] = rho[i] * vec_dot(s + i*n,q,n);
        vec_add(q,q,y+i*n,n,1,-alpha[i]);
        if( i == end)    break;
    }
    //vec_mul(q,n,gamma);
    cgSolver->solve(t,cgPara,q);
    hessianMul->next_sample();
    vec_cpy(q,t,n);
    for(uint32_t i = end; true ; i = (i+1) % m){
        double beta = rho[i] * vec_dot(y+i*n,q,n);
        vec_add(q,q,s+i*n,n,1,alpha[i]-beta);
        if( i == beg)    break;
    }
}


void SampleLBFGS::prepare_optimize(const Problem* data){
    n = data->n;
    numData = data->l;
    Real *mem = new Real[3 * n + numData + 2 * n + 2*n*m + 2*m + 4*n];
    // self memory=3*n[w,g,d] + l
    w = mem;
    std::memset(w,0,sizeof(Real) * n); // init
    g = mem + n;
    d = mem + 2*n;
    // memory=numData +1 [ predict prob, funcVal]
    gradientCalc = new GradientCalc(data);
    gradientCalc->set_memory(mem + 3 * n);
    // memory=2*n[xp,gp]
    linearSearch = new LinearSearch(gradientCalc);
    linearSearch->set_memory(mem + 3*n + numData, mem + 3*n + numData + n);
    s = mem + 3*n + numData + 2*n; //new double[n*m];
    y = s + n*m; //new double[n*m];
    rho = s + 2*n*m; //new double[m];
    alpha = s + 2*n*m + m; //new double[m];
    Real *p = alpha + m;
    hessianMul = new HessianVecProduct(gradientCalc);
   // memory=3*n[r,p,y]
    cgSolver = new CGSolver(hessianMul,n);
    cgSolver->set_memory(p,p + n, p + 2*n);
    t = p + 3*n;
}
void SampleLBFGS::prepare_optimize(GradientCalc *_gradientCalc,const Real *initalPara){
    gradientCalc = _gradientCalc;
    n = gradientCalc->get_parameter_size();
    numData = gradientCalc->get_instance_num();
    Real *mem = new Real[3 * n + numData + 2 * n + 2*n*m + 2*m + 4*n];
    // self memory=3*n[w,g,d] + l
    w = mem;
    if(initalPara == 0){
        std::memset(w,0,sizeof(Real) * n); // init
    }else{
        vec_cpy(w,initalPara,n,1);
    }
    g = mem + n;
    d = mem + 2*n;
    // memory=numData +1 [ predict prob, funcVal]
    gradientCalc->set_memory(mem + 3 * n);
    // memory=2*n[xp,gp]
    linearSearch = new LinearSearch(gradientCalc);
    linearSearch->set_memory(mem + 3*n + numData, mem + 3*n + numData + n);
    s = mem + 3*n + numData + 2*n; //new double[n*m];
    y = s + n*m; //new double[n*m];
    rho = s + 2*n*m; //new double[m];
    alpha = s + 2*n*m + m; //new double[m];
    Real *p = alpha + m;
    hessianMul = new HessianVecProduct(gradientCalc);
   // memory=3*n[r,p,y]
    cgSolver = new CGSolver(hessianMul,n);
    cgSolver->set_memory(p,p + n, p + 2*n);
    t = p + 3*n;
}
void SampleLBFGS::set_sample_parameter(const CGPara& _cgPara,double sampleRatio){
    cgPara = _cgPara;
    if(sampleRatio > 0 && sampleRatio <= 1){
        uint32_t sampleSize = (uint32_t)(gradientCalc->get_instance_num() * sampleRatio);
        if(!hessianMul->set_sample_size(sampleSize)){
            Log::warn("SampleLBFGS::set_sample_parameter","set sampleRatio failed.");
        }
    }else{
        Log::warn("SampleLBFGS::set_sample_parameter","wrong sampleRatio value.");
    }
}
std::string SampleLBFGS::report_algo_para()const{
    std::stringstream ss;
    ss<<"algo="<<optAlgoName<<",m="<<m<<",hessianSampleRatio="<<hessianMul->get_sample_size() / 
        static_cast<double>(numData)<<","<<optimizePara.report()<<",memory="<<(sizeof(Real)*(9*n+numData+2*m+2*n*m)>>30)<<"G";
    return ss.str();
}
