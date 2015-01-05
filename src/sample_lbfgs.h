#ifndef _SAMPLE_LBFGS_H
#define _SAMPLE_LBFGS_H
#include "optimizer.h"
#include "conjugate_gradient.h"
#include "hessian_vec_product.h"
#include "dataset.h"
#include "lbfgs.h"

class SampleLBFGS : public LBFGS{
  public:
    SampleLBFGS(){
        optAlgoName = "SampleLBFGS";
    };
    virtual void prepare_optimize(const Problem* data);
    virtual void prepare_optimize(GradientCalc*,const Real*);
    virtual std::string report_algo_para()const;
    virtual void post_optimize(){
        LBFGS::post_optimize();
        delete hessianMul;
        delete cgSolver;
    };
    void set_sample_parameter(const CGPara& _cgPara,double _sampleRatio);
  protected:
    HessianVecProduct *hessianMul;
    CGSolver *cgSolver;
    CGPara cgPara;
    Real *t;
    virtual void calc_bfgs_direction(uint32_t k,double gamma);
};
#endif
