#include "dataset.h"
#include "optimizer.h"
#include "conjugate_gradient.h"
#include "trust_region.h"

OptimizePara para;
TronPara tronPara;

void config(){
  para.maxIter = 100;
  para.gNormKsi = 1e-12;
  para.intercept = 0;
  para.fitIntercept = false;
  para.l2 = 0;
  para.LS_scaleRate = 0.6;
  para.LS_initStep = 1;
  para.LS_c1 = 5e-5;
  para.BFGS_m = 10;
  para.decRatio = 1e-6;
  para.maxTrainSeconds = 220;


  tronPara.yita1 = 0.25;
  tronPara.yita2 = 0.75;
  tronPara.sigma1 = 0.25;
  tronPara.sigma2 = 0.5;
  tronPara.sigma3 = 4;
  tronPara.sampleRatio = 1;
}
