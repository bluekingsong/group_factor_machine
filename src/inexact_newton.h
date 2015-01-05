#ifndef _INEXACT_NEWTON_H
#define _INEXACT_NEWTON_H
#include "optimizer.h"
#include "conjugate_gradient.h"
#include "gradient_calc.h"
#include "hessian_vec_product.h"

class InexactNewton : public Optimizer {
  public:
    virtual void optimize();
    virtual void prepare_optimize(const Problem* data);
    virtual void post_optimize();
    void set_cgPara(const CGPara& para,double _sampleRatio = 1){
        cgPara = para;
        sampleRatio = _sampleRatio;
        if(hessianMul != 0)    hessianMul->set_sample_size((uint32_t)(_sampleRatio * numData));
    }
    static void unittest();
  protected:
    HessianVecProduct *hessianMul;
    CGSolver *cgSolver;
    CGPara cgPara;
    double sampleRatio;
};

#endif
