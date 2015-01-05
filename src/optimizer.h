#ifndef _OPTIMIZER_H
#define _OPTIMIZER_H
#include <string>
#include <sstream>
#include <ctime>
#include "dataset.h"
#include "linear_search.h"
#include "gradient_calc.h"
#include "cpp_common.h"

struct OptimizePara {
    uint32_t maxIter;
    double gNormKsi;
    double decRatio;  // (f - fNew)/f > decRatio
    double intercept;
    bool fitIntercept;     // if set to true, then the intercept is not used.
    double l2;
    double LS_scaleRate;
    double LS_initStep;
    double LS_c1;
    uint32_t BFGS_m;
    uint32_t maxTrainSeconds;
    std::string report()const{
      std::stringstream ss;
      ss<<"maxIter="<<maxIter<<",maxTrainSecond="<<maxTrainSeconds<<",intercept="<<intercept<<
          ",l2="<<l2<<",LS_scaleRate="<<LS_scaleRate<<",LS_c1="<<LS_c1;
      return ss.str();
    }
};
class Optimizer {
  public:
    virtual void optimize();
    virtual void prepare_optimize(const Problem*)=0;
    virtual void prepare_optimize(GradientCalc*,const Real*)=0;
    virtual void post_optimize();
    virtual std::string report_algo_para()const=0;
    virtual void set_parameter(const OptimizePara& optPara);
    virtual const Real* get_parameter()const{
        return w;
    }
    virtual double get_funcVal()const{
        return funcVal;
    }
  protected:
    OptimizePara optimizePara;
    GradientCalc *gradientCalc;
    LinearSearch *linearSearch;
    uint32_t n; // parameter vector length
    uint32_t numData;
    Real *w;  // current parameter vec
    Real *g; // current gradient vec
    Real *d; // current search direction
    double funcVal;
    double gNorm;
    uint32_t numIter;
    uint32_t numLS; // number of linear search
    uint32_t numAccessData;
    time_t beginTime;
    std::string optAlgoName;
    // member functions
    std::string make_monitor_str()const;
    virtual bool check_stop_condition(double fNew);
    virtual void common_update(uint32_t dataAccessFactor);
};

#endif
