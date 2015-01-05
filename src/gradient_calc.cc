#include <cstring>
#include <cmath>
#include <cstdlib>
#include <assert.h>
#include <iostream>
#include <cstdio>
#include "gradient_calc.h"
#include "vec_op.h"
#include "log.h"

double GradientCalc::operator()(const Real *w, Real * g){
/*
    vec_cpy(g,w,data->get_feature_num(),l2);  // l2 regularization
    if(fitIntercept)    g[data->get_feature_num() - 1] = 0; // don't reguarize the intercept term
    //std::cout<<"GradientCalc:g0Norm="<<vec_dot(g,g,data->get_feature_num())<<" wNorm="<<vec_dot(w,w,data->get_feature_num())<<std::endl;
    double NLL = 0.5 * l2 * vec_dot(w,w,data->get_feature_num());  // fix bug, should count regularization ||W||^2
    if(fitIntercept)    NLL -= 0.5 * l2 * w[data->get_feature_num() - 1] * w[data->get_feature_num() - 1];
    uint32_t dataSize = data->get_instance_num(), start = 0;
    if(sampleEnabled){
        dataSize = sampleSize;
        start = sampleIndex;
    }
    double sampleWeight = data->get_instance_num() / dataSize;
    for(uint32_t i = start; i < start + dataSize && i < data->get_instance_num(); ++i){
        const FeatureNode *instance = data->x[i];
        uint8_t yi = data->y[i];
        double t = fitIntercept ? 0 : intercept;
        int j=0;
        while(-1 != instance[j].index){
            t += w[instance[j].index] * instance[j].value;
            ++j;
        }
        if(fitIntercept)    t += w[data->get_feature_num() - 1];
        double ui = 1.0 / (1.0 + std::exp(-t));
        aucUti.add_instance(ui,yi);
        j=0;
        while(-1 != instance[j].index){
            g[instance[j].index] += sampleWeight * (ui - yi) * instance[j].value;  // gradient, X'*(U-Y)
            ++j;
        }
        if(fitIntercept)    g[data->get_feature_num() - 1 ] += sampleWeight * (ui - yi);
        predictedProb[i] = ui;
        // preventing overflow
        if(yi)    t < -25 ? NLL -= sampleWeight * t : NLL -= sampleWeight * log(ui);
        else      t > 25  ? NLL += sampleWeight * t : NLL -= sampleWeight * log(1-ui);
    }
    funcVal = NLL;
    return NLL;
*/
    return 0.0;
}
GradientCalc* GradientCalc::unittest(const Problem* _data){
/*
 * w[0,1,2,3]=0, w[4]=1 , intercept=0;
 *   x[1],x[2],yi,ti,ui, gi,     g[0], g[1], g[2], g[3], g[4],   ui*(1-ui)
 i=0,  0,  1,  1, 0,0.5,-0.5,    -0.5, -0.5,     ,     ,     ,  0.25
 i=1,  1,  2,  1, 0,0.5,-0.5,        , -0.5, -0.5,     ,     ,  0.25
 i=2,  2,  3,  0, 0,0.5, 0.5,        ,     ,   0.5,  0.5,    ,  0.25
 i=3,  3,  4,  0, 1,s(1),s(1),     ,     ,      , s(1), s(1)  , s(1)*(1-s(1))
 i=4,  4,  0,  0, 1,s(1),s(1), s(1),    ,      ,      , s(1)  , s(1)*(1-s(1))
 i=5,  0,  2,  0, 0,0.5,0.5,       0.5,    ,   0.5,       ,   , 0.25
 i=6,  0,  3,  0, 0,0.5,0.5,       0.5,    ,      ,    0.5,   , 0.25
 i=7,  0,  4,  0, 1,s(1),s(1),    s(1),   ,      ,       , s(1),s(1)*(1-s(1))
 i=8,  1,  3,  0, 0,0.5,0.5,          , 0.5,     ,    0.5,     ,0.25
 i=9,  1,  4,  0, 1,s(1),s(1),       , s(1),    ,       , s(1) ,s(1)*(1-s(1))
 g[0]=0.5+2*s(1)
 g[1]=-0.5+s(1)
 g[2]=0.5
 g[3]=1.5+s(1)
 g[4]=4*s(1)
 a=0.25, b=s(1)*(1-s(1))
 Hessian
    0,    1,   2,    3,   4
0 3a+2b,
1   a,  3a+b
2   a,   a,   3a
3   a,   a,   a,   3a+b
4  2b,   b,   0,    b,   4b
 */
    Log::raw("===========GradientCalc::unittest===============");
    GradientCalc *gradientCalc = new GradientCalc(_data,0);
    uint32_t n = _data->get_feature_num();
    Real *mem = new Real[2 * n + _data->get_instance_num()];
    Real *p = mem + 2 * n;
    gradientCalc->set_memory(p);
    gradientCalc->set_intercept(0,false);
    gradientCalc->set_l2_para(0);
    std::memset(mem,0,sizeof(Real) * n);
    mem[4] = 1.0;
    double f = (*gradientCalc)(mem,mem + n);
    Real *g = mem + n;
    double u = 1.0 / (1.0 + std::exp(-1));
    double required = -6 * std::log(0.5) - 4 * std::log(1 - u);
    //std::cout<<f<<" "<<required<<std::endl;
    assert(std::abs(f - required < 1e-12));
    double t = 1.0 / (1.0 + std::exp(-1));
    double targets[] = { 0.5 + 2 * t, -0.5 + t, 0.5, 1.5 + t, 4 * t }; // 1.9621    0.2311    0.5000    2.2311    2.9242
    for(int i = 0; i < 5; ++i){
        std::cout<<"g[i]="<<g[i]<<" targets[i]="<<targets[i]<<std::endl;
        assert(std::abs(g[i] - targets[i]) < 1e-12);
    }
    double p_target[] = {0.5,0.5,0.5,t,t,0.5,0.5,t,0.5,t};
    for(int i = 0; i < _data->get_instance_num(); ++i){
        assert(std::abs(p_target[i] - p[i]) < 1e-6);
    }
    delete []mem;
    Log::raw("+++++++++++GradientCalc::unittest+++++++++++++++");
    return gradientCalc;
}
