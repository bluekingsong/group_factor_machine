#ifndef _GRADIENT_DESCENT_H
#define _GRADIENT_DESCENT_H
#include <string>
#include "optimizer.h"
#include "dataset.h"

class GradientDescent : public Optimizer{
  public:
    GradientDescent(){
        init();
        optAlgoName = "GradientDescent";
    };
    virtual void optimize();
    virtual void prepare_optimize(GradientCalc*,const Real*);
    virtual std::string report_algo_para()const;
    static double guessInitStep(const Real *g,uint32_t n,uint32_t numIter);
};
#endif
