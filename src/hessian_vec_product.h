#ifndef _HESSIAN_VEC_PRODUCT_H
#define _HESSIAN_VEC_PRODUCT_H
#include "gradient_calc.h"
#include "mat_vec_product.h"

double InstanceDot(const FeatureNode* instance, const Real* v);
class HessianVecProduct : public MatVecProduct {
  protected:
    const GradientCalc* gradientCalc;
    uint32_t sampleSize;
    uint32_t sampleIndex;
  public:
    explicit HessianVecProduct(const GradientCalc* _gradientCalc):gradientCalc(_gradientCalc){
        sampleSize = _gradientCalc->get_data()->l;
        sampleIndex = 0;
    };
    virtual void operator()(const Real *x,const uint32_t n,Real *y)const;
    const GradientCalc* get_gradientCalc()const {
        return gradientCalc;
    }
    bool set_sample_size(uint32_t _sampleSize){
        if(_sampleSize > gradientCalc->get_data()->l)    return false;
        sampleSize = _sampleSize;
        sampleIndex = 0;
        return true;
    }
    uint32_t get_sample_size()const{
        return sampleSize;
    }
    void next_sample();
    static void unittest(GradientCalc *gradientCalc);
};
#endif
