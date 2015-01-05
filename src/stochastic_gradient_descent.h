#ifndef _STOCH_GRADIENT_DESCENT_H
#define _STOCH_GRADIENT_DESCENT_H
#include "optimizer.h"
#include "dataset.h"

class StochGradientDescent : public Optimizer{
  public:
    virtual void optimize();
    virtual void prepare_optimize(const Problem* data);
};
#endif
