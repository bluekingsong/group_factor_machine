#include <cmath>
#include <cstring>
#include "vec_op.h"
#include "online_optimizer.h"
#include "log.h"

DEFINE_double(learn_rate,0.01,"learn rate for online update");
DEFINE_uint64(monitor_iter,10000000,"the monitor(to print some info) iteration");
DEFINE_int32(batch_size,1,"bath size for online training");

void OnlineOptimizer::optimize(){
    //set_parameter(optimizePara);
    gradientCalc->set_sample_para(batchSize);
    gradientCalc->set_sample_enabled(true);
    std::map<uint32_t,double> g_dict;
    numIter = 0;
    numAccessData = 0;
    beginTime = time(0);
    while(numIter < gradientCalc->get_instance_num()){
        ++numIter;
        (*gradientCalc)(w,g_dict);
        gradientCalc->next_sample();
        update(w,g_dict);
        numAccessData += gradientCalc->get_sample_size();
        if(numIter % monitorIter == 0){
            //double train_auc = gradientCalc->auc();
            double logloss = gradientCalc->get_logloss() / numIter;
            double l2loss = gradientCalc->get_l2loss() / numIter;
            std::cout<<"[INF]inner loop,time="<<time(0)-beginTime<<",numIter="<<numIter<<",logloss="<<logloss<<
                ",l2loss="<<l2loss<<std::endl;
            gradientCalc->reset_loss();
        }
    }
}
void OnlineOptimizer::update(Real *w,const std::map<uint32_t,double>& g_dict){
   for(std::map<uint32_t,double>::const_iterator iter = g_dict.begin(); iter != g_dict.end(); ++iter){
       g[iter->first] += iter->second * iter->second;
       //std::cout<<"index="<<iter->first<<" w="<<w[iter->first]<<" g="<<iter->second<<" cg="<<g[iter->first]<<
       //    " new_w="<<w[iter->first]+learnRate * iter->second / std::sqrt(g[iter->first])<<std::endl;
       w[iter->first] -= learnRate * iter->second / std::sqrt(g[iter->first]);
   }
}
void OnlineOptimizer::prepare_optimize(GradientCalc* _gradientCalc,const Real* initW){
    gradientCalc = _gradientCalc;
    n = gradientCalc->get_parameter_size();
    numData = gradientCalc->get_instance_num();
    Real *mem = new Real[2 * n];
    // self memory=3*n[w,g,d] + l
    w = mem;
    if(0 == initW){
        std::memset(w,0,sizeof(Real) * n); // init
    }else{
        vec_cpy(w,initW,n,1);
    }
    g = mem + n;
    std::memset(g,0,sizeof(Real) * n);
    d = 0;
}

std::string OnlineOptimizer::report_algo_para()const{
    std::stringstream ss;
    ss<<"algo="<<optAlgoName<<",learn_rate="<<learnRate;
    return ss.str();
}
