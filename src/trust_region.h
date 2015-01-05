#ifndef _TRUST_REGION_H
#define _TRUST_REGION_H
#include "optimizer.h"
#include "conjugate_gradient.h"
#include "gradient_calc.h"
#include "hessian_vec_product.h"
#include "inexact_newton.h"

struct TronPara{
    double yita1;
    double yita2;
    double sigma1;
    double sigma2;
    double sigma3;
    double sampleRatio;
};
class TRON : public InexactNewton {
  public:
    virtual void optimize();
    void set_para(const CGPara& _cgPara, const TronPara& _tronPara){
        tronPara = _tronPara;
        InexactNewton::set_cgPara(_cgPara,tronPara.sampleRatio);
    }
    double update_trust_region_radius(double)const;
    static void unittest();
  protected:
    TronPara tronPara;
};

#endif
