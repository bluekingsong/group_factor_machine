#ifndef _LINEAR_SEARCH_H
#define _LINEAR_SEARCH_H
#include "gradient_calc.h"

// default it is backtracking linear search, if you want to 
// implement other linear search method, just inherit it
// search for Xi+1 = X + alpha * (dFactor * D)
class LinearSearch{
  public:
    // x:current parameter,g:current gradient,d:search direction dFactor: scale adjustment of d
    virtual double operator() (const Real *x,const Real *g, const Real *d, double dFactor = 1);
    explicit LinearSearch(GradientCalc *_gradientCalc):gradientCalc(_gradientCalc){
        scaleRatio = 0.7;
        initStep = 1;
        c1 = 0.2;
        n = _gradientCalc->get_parameter_size();
    }
    void setPara(double _scaleRatio,double _initStep, double _c1){
        scaleRatio = _scaleRatio;
        initStep = _initStep;
        c1 = _c1;
    }
    void set_init_step(double a){
        initStep = a;
    }
    double get_init_step()const{
        return initStep;
    }
    void set_memory(Real *_xp, Real *_gp){
            xp = _xp;
            gp = _gp;
    }
    const Real* get_new_parameter()const{
        return xp;
    }
    const Real* get_new_gradient()const{
        return gp;
    }
    double get_new_funcVal()const{
        return funcVal;
    }
    uint32_t get_num_search()const{
        return numIter;
    }
    static void unittest(GradientCalc *gradientCalc);
  protected:
    uint32_t n;
    uint32_t numIter;
    GradientCalc* gradientCalc;
    Real *xp;
    Real *gp;
    double initStep;
    double scaleRatio;
    double c1;
    double funcVal;
};
#endif
