#include <cmath>
#include <iostream>
#include <cstring>
#include "assert.h"
#include "conjugate_gradient.h"
#include "vec_op.h"
#include "log.h"
#include "cpp_common.h"

void CGSolver::solve(Real *x, const CGPara& para,const Real *b, double bFactor){
  assert(para.maxIter != 0);
  std::memset(x,0,sizeof(Real) * n);
  vec_cpy(r,b,n,-bFactor);
  vec_cpy(p,b,n,bFactor);
  //vec_mul(p,n,-1);
  uint32_t k = 0;
  double rNorm = vec_dot(r,r,n);
  double xNorm = 0;
  double bNorm = bFactor * bFactor * vec_dot(b,b,n);
  double ksi = para.errorNormKsi;
  if(para.autoNormKsi){
      double sqrtBNorm = sqrt(sqrt(bNorm));
      ksi = sqrtBNorm > 0.5 ? 0.5 : sqrtBNorm;
  }
  //std::cout<<"rNorm="<<rNorm<<std::endl;
  while( k++ < para.maxIter  && rNorm > ksi * ksi * bNorm){
      (*matVecProduct)(p,n,y);
      double t = vec_dot(p,y,n);
      if(para.checkPositiveDefined && t <= 0){
          Log::info("CGSolver::solve","positive define of coefficent martix not satified.");
          if(1 == k)    vec_cpy(x,b,n,bFactor);
      }
      double alpha = rNorm / t;
      if( std::abs(alpha) < para.zeroEps ){
          Log::warn("CGSolver::solve","break on a 0 step size,iter=" + CppCommonFunction::StringFunction::to_string(k) + "/" + CppCommonFunction::StringFunction::to_string(para.maxIter));
          break;
      }
      if(para.xNormKsi > 0){
          double xDotp = vec_dot(x,p,n);
          double pNorm = vec_dot(p,p,n);
          double xNextNorm = xNorm + 2 * alpha * xDotp + alpha * alpha * pNorm;
          if(xNextNorm >= para.xNormKsi * para.xNormKsi){  // encounter trust region radius boundary
              double a = pNorm, b = 2 * xDotp, c = xNorm - para.xNormKsi * para.xNormKsi;
              double delta = b * b - 4 * a * c;
              if(delta < 0){
                  Log::warn("CGSolver::solve","no solution for trust region radius boundary");
                  break;
              }else{
                  double t = (-b + std::sqrt(delta)) / (2 * a);  // TODO: or -b -sqrt(delta) ? 
                  vec_add(x,x,p,n,1,t);
                  break;
              }
          }
          xNorm = xNextNorm;
      }
      vec_add(x, x, p, n, 1, alpha);
      vec_add(r, r, y, n, 1, alpha);
      double rNormNew = vec_dot(r,r,n);
      double beta =  rNormNew / rNorm;
      rNorm = rNormNew;
      vec_add(p, r, p, n, -1.0, beta);
  }
  numIter = k;
};
void CGSolver::unittest(const HessianVecProduct *hessianMul){
    Log::raw("===========CGSolver::unittest===============");
    const GradientCalc *gradientCalc = hessianMul->get_gradientCalc();
    uint32_t n = gradientCalc->get_parameter_size();
    CGSolver *cgSolver = new CGSolver(hessianMul,n);
    Real *mem = new Real[6 * n];
    std::memset(mem,0,sizeof(Real) * 5 * n);
    cgSolver->set_memory(mem,mem + n, mem + 2 * n);
    Real *w = mem + 3 * n;
    w[4] = 1;
    Real *g = mem + 4 * n;
    Real *d = mem + 5 * n;
    double f = (*const_cast<GradientCalc*>(gradientCalc))(w,g);
    CGPara para;
    para.maxIter = 10; para.errorNormKsi = 1e-12; para.zeroEps = 1e-12,para.checkPositiveDefined = true, para.autoNormKsi = true;
    for(int i = 1; i < 50; ++i){
        cgSolver->solve(d,para,g,-1);
        double dec = vec_dot(g,d,n);
        assert(dec < 0);
    }
    delete []mem;
    delete cgSolver;
    Log::raw("+++++++++++CGSolver::unittest+++++++++++++++");
}

