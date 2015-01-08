#include <iostream>
#include <sstream>
#include <ctime>
#include <cmath>
#include "optimizer.h"
#include "vec_op.h"
#include "log.h"

DEFINE_bool(fitIntercept,false,"fitIntercept?");
DEFINE_double(Intercept,0,"intercept,if fitIntercept set to true,this is value has no effect");
DEFINE_double(l2,1e5,"l2");

std::string Optimizer::make_monitor_str() const{
    std::stringstream ss;
    time_t now = time(0);
    double wNorm = vec_dot(w,w,n);
    if(optimizePara.fitIntercept)   wNorm -= w[n - 1] * w[n - 1];
    ss<<"iter="<<numIter<<" time="<<now - beginTime<<" numData="<<numAccessData<<" funcVal="<<
        funcVal<<" gNorm="<<gNorm<<" squareWNorm="<<wNorm<<" numLS="<<numLS;
    if(optimizePara.fitIntercept)   ss<<" intercept="<<w[n-1];
    return ss.str();
};
void Optimizer::set_parameter(const OptimizePara& optPara){
    optimizePara = optPara;
    beginTime = time(0);
    numLS = numAccessData = 0;
    if(gradientCalc != 0){
        gradientCalc->set_intercept(optPara.intercept,optPara.fitIntercept);
        gradientCalc->set_l2_para(optPara.l2);
    }
    if(linearSearch != 0){
        linearSearch->setPara(optPara.LS_scaleRate,optPara.LS_initStep,optPara.LS_c1);
    }
}
void Optimizer::optimize(){
    set_parameter(optimizePara);
    std::cout<<"[Algo Para]:"<<",parameter_vec_size="<<n<<","<<report_algo_para()<<std::endl<<std::flush;
    funcVal = (*gradientCalc)(w,g);
    numIter = 0;
    gNorm = std::sqrt(vec_dot(g,g,n));
}
void Optimizer::common_update(uint32_t dataAccessFactor){
    
}
void Optimizer::post_optimize(){
    std::cout<<"[Report]:"<<make_monitor_str()<<std::endl<<std::flush;
    if(linearSearch)    delete linearSearch;
    if(w)    delete []w;
}
bool Optimizer::check_stop_condition(double fNew){
  ++numIter;
  double _funcVal = funcVal;
  funcVal = linearSearch->get_new_funcVal();
  if(numIter < 3)    return false;
  double decRatio = (_funcVal - fNew) / _funcVal;
  time_t now = time(0);
  if(numIter < optimizePara.maxIter && gNorm > optimizePara.gNormKsi
     && decRatio > optimizePara.decRatio && (now - beginTime) <= optimizePara.maxTrainSeconds){
      return false;
  }
  std::string info = make_monitor_str();
  //Log::raw(info);
  std::cerr<<info<<std::endl<<std::flush;
  return true;
}
