#ifndef _ONLINE_OPTIMIZER_H
#define _ONLINE_OPTIMIZER_H
#include "optimizer.h"
#include <gflags/gflags.h>

DECLARE_double(learn_rate);
DECLARE_uint64(monitor_iter);
DECLARE_int32(batch_size);

class OnlineOptimizer:public Optimizer{
  public:
    OnlineOptimizer(){
        init();
        optAlgoName = "SGD";
    }
    virtual void init(){
        Optimizer::init();
        learnRate = FLAGS_learn_rate;
        monitorIter = FLAGS_monitor_iter;
        batchSize = FLAGS_batch_size;
    }
    virtual void optimize();
    virtual void prepare_optimize(GradientCalc*,const Real*);
    virtual void post_optimize(){    if(w)    delete[] w; }
    virtual std::string report_algo_para()const;
    void set_online_parameter(double _learn_rate,uint64_t _monitor_iter,uint32_t _batchSize){
        learnRate = _learn_rate;
        monitorIter = _monitor_iter;
        batchSize = _batchSize;
    }
  protected:
    double learnRate;
    uint64_t monitorIter;
    uint32_t batchSize;
    void update(Real *w,const std::map<uint32_t,double>& g_dict);
};

#endif
