#include <cstring>
#include <iostream>
#include "linear_search.h"
#include "gradient_descent.h"
#include "vec_op.h"
#include "lbfgs.h"
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
void LBFGS::calc_bfgs_direction(uint32_t k,double gamma){
    double *q = d;
    vec_cpy(q,g,n);
    uint32_t beg = (k-1) % m, end = 0;
    if( k > m)    end = k % m;
    for(uint32_t i = beg; true ; i = (m+i-1) % m){
        alpha[i] = rho[i] * vec_dot(s + i*n,q,n);
        vec_add(q,q,y+i*n,n,1,-alpha[i]);
        if( i == end)    break;
    }
    vec_mul(q,n,gamma);
    for(uint32_t i = end; true ; i = (i+1) % m){
        double beta = rho[i] * vec_dot(y+i*n,q,n);
        vec_add(q,q,s+i*n,n,1,alpha[i]-beta);
        if( i == beg)    break;
    }
}
void LBFGS::optimize(){
    Optimizer::optimize();
    while(true){
        uint32_t k = numIter;
        double step_len;
        if( 0 == k ){
            linearSearch->set_init_step(GradientDescent::guessInitStep(g,n,numIter));
            step_len = (*linearSearch)(w,g,g,-1);
            linearSearch->set_init_step(optimizePara.LS_initStep);
        }
        else{
            double t1 = vec_dot(y+(k-1)%m*n,y+(k-1)%m*n,n);
            double t2 = vec_dot(s+(k-1)%m*n,y+(k-1)%m*n,n);
            if(t2 < 0){
              Log::error("LBFGS","we get Sk*Yk < 0, is the object function not strongly convex while Wolfe condition is not satisfied? see Numerical Optimization(p195)");
              break;
            }
            double gamma = t2 / t1;
            if( std::abs(gamma) < 1e-12 ){
                Log::warn("LBFGS","zero init inverse-Hessian approxmiation produced.");
                break;
            }
            //std::cout<<"k="<<k<<" gamma="<<gamma<<" t1="<<t1<<" t2="<<t2<<std::endl;
            calc_bfgs_direction(k,gamma);
            step_len = (*linearSearch)(w,g,d,-1);
        }
        if( step_len < 0 ){
            Log::error("LBFGS::do_optimize","can't find suitable step size at line-search");
            break;
        }
        uint32_t numSearch = linearSearch->get_num_search();
        numLS += numSearch;
        numAccessData += numSearch * numData;
        const Real *xp = linearSearch->get_new_parameter();
        const Real *gp = linearSearch->get_new_gradient();
        vec_add(s+n*(k%m),xp,w,n,1,-1);
        vec_add(y+n*(k%m),gp,g,n,1,-1);
        rho[k%m] = 1.0 / vec_dot(y+n*(k%m),s+n*(k%m),n);
        vec_cpy(w,xp,n);
        vec_cpy(g,gp,n);
        if(check_stop_condition(linearSearch->get_new_funcVal())){
            break;
        }
        funcVal = linearSearch->get_new_funcVal();
        gNorm = std::sqrt(vec_dot(g,g,n));
        Log::raw(make_monitor_str());
    }
}
void LBFGS::prepare_optimize(GradientCalc *_gradientCalc,const Real *initalPara){
    gradientCalc = _gradientCalc;
    n = gradientCalc->get_parameter_size();
    numData = gradientCalc->get_instance_num();
    Real *mem = new Real[3 * n + numData + 2 * n + 2*n*m + 2*m];
    // self memory=3*n[w,g,d] + l
    w = mem;
    if(0 == initalPara){
        std::memset(w,0,sizeof(Real) * n); // default init is set to 0
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
}

void LBFGS::set_parameter(const OptimizePara& optPara){
    Optimizer::set_parameter(optPara);
    m = optPara.BFGS_m;
}
std::string LBFGS::report_algo_para()const{
    std::stringstream ss;
    ss<<"algo="<<optAlgoName<<",m="<<m<<","<<optimizePara.report()<<",memory="<<
        (sizeof(Real)*(5*n+numData+2*n*m+2*m)>>30)<<"G";
    return ss.str();
}
