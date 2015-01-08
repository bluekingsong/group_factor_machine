#ifndef _LBFGS_H
#define _LBFGS_H
#include <string>
#include "optimizer.h"
#include "dataset.h"

class LBFGS : public Optimizer{
  public:
    LBFGS(){
        init();
        optAlgoName = "LBFGS";
    };
    virtual void init(){ Optimizer::init(); s = y = rho = alpha = 0;  } //TODO
    virtual void optimize();
    virtual void prepare_optimize(GradientCalc*,const Real*);
    virtual void set_parameter(const OptimizePara& optPara);
    virtual std::string report_algo_para()const;
    virtual void post_optimize(){
        Optimizer::post_optimize();
    }
  protected:
    Real *s; // delta(x)
    Real *y; // delta(g);
    Real *rho;
    Real *alpha;
    uint32_t m;
    virtual void calc_bfgs_direction(uint32_t k,double gamma);
};
#endif
