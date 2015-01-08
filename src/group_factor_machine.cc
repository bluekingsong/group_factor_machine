#include <iostream>
#include <cstdlib>
#include <ctime>
#include "group_factor_machine.h"
#include "gradient_descent.h"
#include "gfm_gradient_calc.h"
#include "vec_op.h"
#include "lbfgs.h"
#include "online_optimizer.h"
//#include "sample_lbfgs.h"
DEFINE_double(epsilon,1e-6,"the epsilon for computting lasso gradient");
DEFINE_double(intercept,0,"the intercept for logistic function");
DEFINE_string(optName,"sgd","the optimize name");
DEFINE_uint64(groupMaxIter,10,"the max iteration in each grop optimization");
DEFINE_double(init_stdv,1e-4,"the stdv for parameter init");
DEFINE_string(train_file,"data/train.dat","train data file");
DEFINE_string(test_file,"data/train.dat","test data file");
DEFINE_double(groupLambda,0.01,"grop lasso lambda");
DEFINE_uint64(maxIter,10," max out iteration");

void GroupFactorMachine::train(const std::string& optimizerName,uint64_t groupMaxIter){
    Optimizer * optimizer;
    //CGPara cgPara;
    //double hessianSampleRatio = 0.01;
    if(optimizerName == "gd"){
       optimizer = new GradientDescent();
    }else if(optimizerName == "sgd"){
        groupMaxIter = dataset->get_instance_num() / FLAGS_batch_size;
        optimizer = new OnlineOptimizer();
        ((OnlineOptimizer*)optimizer)->set_online_parameter(FLAGS_learn_rate,FLAGS_monitor_iter,FLAGS_batch_size);
    }else if(optimizerName == "bfgs"){
       optimizer = new LBFGS();
    }else /*if(optimizerName == "sbfgs"){
        optimizer = new SampleLBFGS();
        cgPara.maxIter = 8, cgPara.zeroEps = 1e-8, cgPara.errorNormKsi = 0.1;
        cgPara.autoNormKsi = true, cgPara.checkPositiveDefined = true;
        cgPara.xNormKsi = -1; // no effect
    }else*/{
        std::cerr<<"[ERR] unsupported optimizer given."<<std::endl;
        return;
    }
    init_parameter();
    GfmGradientCalc *gradientCalc = new GfmGradientCalc(dataset,intercept);
    gradientCalc->set_epsilon(FLAGS_epsilon);
    gradientCalc->set_group_lambda(FLAGS_groupLambda);
    gradientCalc->set_intercept(FLAGS_Intercept,FLAGS_fitIntercept);
    gradientCalc->set_l2_para(FLAGS_l2);
    gradientCalc->init_factor_info(parameter,latentSpaceDimension,groupDimension);
    gradientCalc->reset_loss();
    double funcObj = -1;
    uint32_t iteration = 0;
    uint32_t contextNumFeature = groupDimension[0];
    uint32_t maxNumFeature = 0;
    for(uint32_t group = 1; group < groupDimension.size(); ++group){
        if(groupDimension[group] > maxNumFeature){
            maxNumFeature = groupDimension[group];
        }
    }
    AucUti& aucUti = gradientCalc->get_auc_uti();
    Real *partialW = new Real[contextNumFeature + maxNumFeature * latentSpaceDimension];
    time_t beginTime = time(0);
    for(;iteration < FLAGS_maxIter; ++iteration){
        uint64_t groupOffset = contextNumFeature;
        double progress_auc = 0.0;
        double progress_loss = 0.0;
        for(uint32_t group = 1; group < groupDimension.size(); ++group){
            gradientCalc->set_active_group(group);
            gradientCalc->clear_auc();
            //std::cerr<<"[INF]"<<"active_group="<<group<<"  parameter vec size="<<gradientCalc->get_parameter_size()<<std::endl<<std::flush;
            vec_cpy(partialW,parameter,contextNumFeature,1);
            vec_cpy(partialW + contextNumFeature,parameter + groupOffset,
                    latentSpaceDimension * groupDimension[group],1);
            //para.maxIter = groupMaxIter;
            //optimizer->set_parameter(para);
            optimizer->prepare_optimize(gradientCalc,partialW);
            if(optimizerName == "sbfgs"){
              //  ((SampleLBFGS*)optimizer)->set_sample_parameter(cgPara,hessianSampleRatio);
            }
            //optimizer->set_parameter(para);
            optimizer->optimize();
            const Real *p = optimizer->get_parameter();
            vec_cpy(parameter,p,contextNumFeature,1);
            vec_cpy(parameter + groupOffset,p + contextNumFeature,
                    latentSpaceDimension * groupDimension[group],1);
            progress_auc = aucUti.auc();
            progress_loss = aucUti.logloss() / dataset->get_instance_num();
            groupOffset += groupDimension[group] * latentSpaceDimension;
            funcObj = optimizer->get_funcVal();
            optimizer->post_optimize();
            gradientCalc->reset_sample_index();
            std::cout<<"[INF]iter="<<iteration<<" active_group="<<group<<" progress-auc="<<progress_auc<<" progress-logloss="<<progress_loss<<std::endl;
        }
        gradientCalc->test_data(*test_data);
        double test_auc = aucUti.auc();
        double test_logloss = aucUti.logloss() / test_data->get_instance_num();
        std::cerr<<"[GroupFactorMachine::train]"<<"time="<<time(0) - beginTime<<" iter="<<iteration<<" train-auc="<<progress_auc<<" test-auc="<<test_auc<<
            " train-logloss="<<progress_loss<<" test-logloss="<<test_logloss<<std::endl<<std::endl<<std::flush;
        //dataset->shuffle();
    }
}
void GroupFactorMachine::init_parameter(){
    uint64_t size = groupDimension[0];
    for(uint32_t group = 1; group < groupDimension.size(); ++group){
        size += latentSpaceDimension * groupDimension[group];
    }
    parameter = new Real[size];
    for(uint64_t i = 0; i < size; ++i){
        parameter[i] = (std::rand() / (double)RAND_MAX - 0.5) * FLAGS_init_stdv;
    }
}
