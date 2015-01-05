#include <iostream>
#include <cstdlib>
#include "group_factor_machine.h"
#include "gradient_descent.h"
#include "gfm_gradient_calc.h"
#include "config.h"
#include "vec_op.h"
#include "lbfgs.h"
#include "sample_lbfgs.h"

void GroupFactorMachine::train(const std::string& optimizerName,uint32_t groupMaxIter){
    Optimizer * optimizer;
    CGPara cgPara;
    double hessianSampleRatio = 0.01;
    if(optimizerName == "gd"){
       optimizer = new GradientDescent();
    }else if(optimizerName == "bfgs"){
       optimizer = new LBFGS();
    }else if(optimizerName == "sbfgs"){
        optimizer = new SampleLBFGS();
        cgPara.maxIter = 8, cgPara.zeroEps = 1e-8, cgPara.errorNormKsi = 0.1;
        cgPara.autoNormKsi = true, cgPara.checkPositiveDefined = true;
        cgPara.xNormKsi = -1; // no effect
    }else{
        std::cerr<<"[ERR] unsupported optimizer given."<<std::endl;
        return;
    }
    init_parameter();
    GfmGradientCalc *gradientCalc = new GfmGradientCalc(dataset,intercept);
    gradientCalc->set_epsilon(epsilon);
    gradientCalc->set_group_lambda(0.01);
    gradientCalc->init_factor_info(parameter,latentSpaceDimension,groupDimension);
    double funcObj = -1;
    uint32_t iteration = 0;
    uint32_t contextNumFeature = groupDimension[0];
    uint32_t maxNumFeature = 0;
    for(uint32_t group = 1; group < groupDimension.size(); ++group){
        if(groupDimension[group] > maxNumFeature){
            maxNumFeature = groupDimension[group];
        }
    }
    double auc;
    Real *partialW = new Real[contextNumFeature + maxNumFeature * latentSpaceDimension];
    for(;true; ++iteration){
        uint64_t groupOffset = contextNumFeature;
        for(uint32_t group = 1; group < groupDimension.size(); ++group){
            gradientCalc->set_active_group(group);
            std::cerr<<"[INF] parameter vec size="<<gradientCalc->get_parameter_size()<<std::endl<<std::flush;
            vec_cpy(partialW,parameter,contextNumFeature,1);
            vec_cpy(partialW + contextNumFeature,parameter + groupOffset,
                    latentSpaceDimension * groupDimension[group],1);
            config();
            para.maxIter = groupMaxIter;
            optimizer->set_parameter(para);
            optimizer->prepare_optimize(gradientCalc,partialW);
            if(optimizerName == "sbfgs"){
                ((SampleLBFGS*)optimizer)->set_sample_parameter(cgPara,hessianSampleRatio);
            }
            optimizer->set_parameter(para);
            optimizer->optimize();
            const Real *p = optimizer->get_parameter();
            vec_cpy(parameter,p,contextNumFeature,1);
            vec_cpy(parameter + groupOffset,p + contextNumFeature,
                    latentSpaceDimension * groupDimension[group],1);
            groupOffset += groupDimension[group] * latentSpaceDimension;
            funcObj = optimizer->get_funcVal();
            auc = gradientCalc->auc();
            gradientCalc->clear_auc();
            optimizer->post_optimize();
            std::cout<<"[INF] logloss="<<gradientCalc->get_logloss()<<" groupLassoLoss="<<gradientCalc->get_group_lasso_loss()<<
                " l2loss="<<gradientCalc->get_l2loss()<<std::endl;
        }
        double test_auc = gradientCalc->calc_test_auc(*test_data);
        std::cerr<<"[GroupFactorMachine::train] iter="<<iteration<<" funcObj="<<funcObj<<
            " train-auc="<<auc<<" test-auc="<<test_auc<<std::endl<<std::flush;
    }
}
void GroupFactorMachine::init_parameter(){
    uint64_t size = groupDimension[0];
    for(uint32_t group = 1; group < groupDimension.size(); ++group){
        size += latentSpaceDimension * groupDimension[group];
    }
    parameter = new Real[size];
    for(uint64_t i = 0; i < size; ++i){
        parameter[i] = std::rand() / INT_MAX / 100 - 0.005;
    }
}
