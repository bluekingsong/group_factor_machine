#ifndef _GRADIENT_CALC_H
#define _GRADIENT_CALC_H
#include "dataset.h"
#include "auc.h"

typedef double Real;
typedef unsigned int uint32_t;

// the Object Function is f(w)=sum(fi(w,xi))/data.l + 0.5*l2*||w||^2
class GradientCalc {
  protected:
    const Problem* data;
    Real *predictedProb;
    double funcVal;
    double intercept;
    bool fitIntercept;  // if fitIntercept = true, then the w[data.n] is the intercept
    double l2;
    uint32_t sampleSize;
    uint32_t sampleIndex;
    bool sampleEnabled; // is in sample mode ?
    AucUti aucUti;
  public:
    GradientCalc(const Problem* _data,double _intercept = 0) : data(_data),intercept(_intercept){
        l2 = 0;
        fitIntercept = false;
        sampleSize = _data->l;
        sampleIndex = 0;
        sampleEnabled= false;
    }
    virtual uint32_t get_parameter_size()const{
        return data->get_feature_num();
    }
    virtual uint32_t get_instance_num()const{
        return data->get_instance_num();
    }
    void next_sample(){
        sampleIndex = (sampleIndex + sampleSize) % data->l;
    }
    void set_sample_enabled(bool flag){
        sampleEnabled = flag;
    }
    bool set_sample_para(uint32_t _sampleSize){
        if(_sampleSize > data->l)    return false;
        sampleSize = _sampleSize;
        return true;
    }
    bool set_sample_para(double sampleRatio){
        if(sampleRatio > 1 || sampleRatio <= 0)   return false;
        sampleSize = (uint32_t)(sampleRatio * data->l);
        return true;
    }
    void set_intercept(double _intercept,bool _fitIntercept){
        intercept = _intercept;
        fitIntercept = _fitIntercept;
    };
    void set_l2_para(double _l2){
        l2 = _l2;
    }
    double get_l2_para()const{
      return l2;
    }
    void set_memory(Real *mem){
        predictedProb = mem;
    };
    void clear_auc(){
        aucUti.clear();
    }
    double auc(){
        return aucUti.auc();
    }
    virtual const FeatureNode *get_instance(uint32_t index)const{
        return data->x[index];
    }
    virtual double operator()(const Real *w, Real * g);
    double get_prediction(uint32_t index)const{
       return predictedProb[index];
    };
    uint32_t get_label(uint32_t index)const{
        return data->y[index];
    }
    double get_funcVal()const{
        return funcVal;
    };
    const Problem* get_data()const{
        return data;
    }
    static GradientCalc* unittest(const Problem* _data);
};
#endif
