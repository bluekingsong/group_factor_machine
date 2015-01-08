#include <cstring>
#include <cmath>
#include <cstdlib>
#include <assert.h>
#include <iostream>
#include <cstdio>
#include "gfm_gradient_calc.h"
#include "vec_op.h"
#include "log.h"

DEFINE_bool(pos_weighting,false," use position bias weighting?");

double GfmGradientCalc::operator()(const Real *w, std::vector<uint32_t>& g_index, double& g_val){
    assert(sampleEnabled);
    assert(sampleSize == 1);
    return 0;
}
double GfmGradientCalc::operator()(const Real *w, std::map<uint32_t,double>& g_dict){
    assert(sampleEnabled);  // only work in sample mode
    groupLassoLoss = -1; // groupLasso don't work in sample mode
    g_dict.clear();
    uint64_t dataSize = sampleSize;
    uint64_t start = sampleIndex;
    for(uint64_t i = start; i < start + dataSize && i < data->l; ++i){
        const FeatureNode *instance = data->x[i];
        uint8_t yi = data->y[i];
        double weight = data->weight[i];
        if(!FLAGS_pos_weighting)    weight = 1.0;
        double t = GfmGradientCalc::predict_signal(instance,parameter,groupOffset,activeGroup,groupSum,latentSpaceDimension,w);
        t += intercept;
        double ui = 1.0 / (1.0 + std::exp(-t));
        aucUti.add_instance(t,yi);
        // calculate temp factor vector
        std::memset(partialSum,0,sizeof(Real) * groupOffset.size() * latentSpaceDimension);
        for(uint32_t j = 1; j < groupOffset.size(); ++j){
            vec_add(partialSum + latentSpaceDimension*j,groupSum,groupSum + latentSpaceDimension*j,latentSpaceDimension,1,-1);
        }
        while(!instance->is_end()){
            uint32_t groupId = instance->get_group_id();
            uint32_t featureId = instance->get_feature_id();
            if(0 == groupId){
                if(g_dict.find(featureId) == g_dict.end()){
                    g_dict[featureId] = weight * (ui - yi) / dataSize  + l2 * w[featureId] / get_parameter_size() ;
                    l2loss += 0.5 * l2 * w[featureId] * w[featureId] / get_parameter_size();
                }
                else    g_dict[featureId] += weight * (ui - yi) / dataSize;
            }else if(groupId == activeGroup){
                uint64_t rowOffset = groupDimension[0] + featureId * latentSpaceDimension;
                for(uint32_t k = 0; k < latentSpaceDimension; ++k){
                    uint64_t offset = rowOffset + k;
                    double g = weight * (ui - yi) * partialSum[groupId * latentSpaceDimension + k] / dataSize;
                    if(g_dict.find(offset) == g_dict.end()){
                        g_dict[offset] = g + l2 * w[offset] / get_parameter_size();
                        l2loss +=  0.5 * l2 * w[offset] * w[offset] / get_parameter_size();
                    }
                    else    g_dict[offset] += g;
                }
            }
            ++instance;
        }
        logloss += AucUti::LogLoss(t,yi);
        //std::cout<<"logloss="<<t_loss<<std::endl;
    }
    return l2loss + logloss;
}

double GfmGradientCalc::operator()(const Real *w, Real * g){
    assert(!fitIntercept);  // don't support fitIntercept
    uint32_t n = this->get_parameter_size();
    std::memset(g,0,sizeof(Real) * n);
    vec_cpy(g,w,groupDimension[0],l2);
    l2loss =  0.5 * vec_dot(w,w,groupDimension[0]);
    uint64_t dataSize = data->l, start = 0;
    if(sampleEnabled){
        dataSize = sampleSize;
        start = sampleIndex;
    }
    logloss = 0.0;
    double sampleWeight = data->l / dataSize;
    for(uint64_t i = start; i < start + dataSize && i < data->l; ++i){
        const FeatureNode *instance = data->x[i];
        uint8_t yi = data->y[i];
        double weight = data->weight[i];
        if(!FLAGS_pos_weighting)    weight = 1.0;
        double t = GfmGradientCalc::predict_signal(instance,parameter,groupOffset,activeGroup,groupSum,latentSpaceDimension,w);
        /*for(uint32_t j = 0; j < groupDimension.size(); ++j){
            vec_print(groupSum + j*latentSpaceDimension,latentSpaceDimension,std::cout,"groupSum");
        }*/
        t += intercept;
        double ui = 1.0 / (1.0 + std::exp(-t));
        aucUti.add_instance(t,yi);
        // calculate temp factor vector
        std::memset(partialSum,0,sizeof(Real) * groupOffset.size() * latentSpaceDimension);
        for(uint32_t j = 1; j < groupOffset.size(); ++j){
            vec_add(partialSum + latentSpaceDimension*j,groupSum,groupSum + latentSpaceDimension*j,latentSpaceDimension,1,-1);
        }
        while(!instance->is_end()){
            uint32_t groupId = instance->get_group_id();
            uint32_t featureId = instance->get_feature_id();
            if(0 == groupId){
                g[featureId] += weight * sampleWeight * (ui - yi); // * instance[j].value; // gradient, X'*(U-Y)
            }else if(groupId == activeGroup){
                uint64_t rowOffset = groupDimension[0] + featureId * latentSpaceDimension;
                for(uint32_t k = 0; k < latentSpaceDimension; ++k){
                    uint64_t offset = rowOffset + k;
                    g[offset] += weight * sampleWeight * (ui - yi) * partialSum[groupId * latentSpaceDimension + k]; // * instance[j].value
                }
            }
            ++instance;
        }
        if(predictedProb)    predictedProb[i] = ui;
        logloss += AucUti::LogLoss(t,yi);
    }
    groupLassoLoss = 0;
    for(uint32_t group = 1; group < groupDimension.size(); ++group){
        for(uint32_t featureId = 0; featureId < groupDimension[group]; ++featureId){
            uint64_t offset = GfmGradientCalc::calc_offset(groupOffset,group,featureId,latentSpaceDimension);
            const Real *p = parameter + offset;
            uint64_t rowOffset = groupDimension[0] + featureId * latentSpaceDimension;
            if( group == activeGroup){
                p = w + rowOffset;
            }
            double sqrNormOfGroup = vec_dot(p,p,latentSpaceDimension);
            double sqrt = std::sqrt(sqrNormOfGroup + epsilon);
            groupLassoLoss += groupLambda * sqrt;
            if(group == activeGroup){
                for(uint32_t k = 0; k < latentSpaceDimension; ++k){
                    g[rowOffset + k] += groupLambda * p[k] / sqrt;
                }
            }
        }
    }
    funcVal = l2loss + logloss + groupLassoLoss; 
    return funcVal;
}
AucUti& GfmGradientCalc::test_data(const Problem& dataset){
    aucUti.clear();
    for(uint64_t i = 0; i < dataset.l; ++i){
        const FeatureNode *instance = dataset.x[i];
        double t = GfmGradientCalc::predict_signal(instance,parameter,groupOffset,0,groupSum,latentSpaceDimension,0);
        t += intercept;
        aucUti.add_instance(t,dataset.y[i]);
    }
    return get_auc_uti();
}
double GfmGradientCalc::predict_signal(const FeatureNode *instance,const Real *wholeParameter,const std::vector<uint64_t>& groupOffset,
                                uint32_t activeGroupId, Real *factor_sum,uint32_t k,const Real *w){
    std::memset(factor_sum,0,sizeof(Real) * k * groupOffset.size());
    double t = 0;
    while( !instance->is_end()){
        uint32_t groupId = instance->get_group_id();
        assert(groupId < groupOffset.size());
        uint32_t featureId = instance->get_feature_id();
        if(0 == groupId){ // context(bias) features, no conjunction
            if(0 == activeGroupId){ // w is not used,the wholeParameter is fixed
                t += wholeParameter[featureId];
            }else{
                t += w[featureId];
            }
        }else{
            uint64_t offset = GfmGradientCalc::calc_offset(groupOffset,groupId,featureId,k);
            const Real *p = wholeParameter + offset;
            if(groupId == activeGroupId){
                uint64_t rowOffset = groupOffset[1] + featureId * k;
                p = w + rowOffset;
            }
            vec_add(factor_sum,factor_sum,p,k,1,1);   // C1 + C2 + C3
            vec_add(factor_sum + k*groupId,factor_sum + k*groupId,p,k,1,1); // Ci for group i
        }
        ++instance;
    }
    t += vec_dot(factor_sum,factor_sum,k);
    for(uint32_t j = 1; j < groupOffset.size(); ++j){
        t -= vec_dot(factor_sum + k * j,factor_sum + k * j,k);
    }
    return 0.5 * t;
}

std::vector<uint64_t> GfmGradientCalc::construct_group_offset(const std::vector<uint32_t>& groupDimension,uint32_t latentSpaceDimension){
    std::vector<uint64_t> groupOffset;
    groupOffset.push_back(0);
    groupOffset.push_back(groupDimension[0]);
    for(uint32_t i = 2; i < groupDimension.size(); ++i){
        groupOffset.push_back(groupOffset[i-1] + latentSpaceDimension * groupDimension[i-1]);
    }
    return groupOffset;
}
uint64_t GfmGradientCalc::calc_offset(const std::vector<uint64_t>& groupOffset,uint32_t group,
                                      uint32_t featureId,uint32_t latentSpaceDimension){
    if(0 == group)    return featureId;
    uint64_t offset = groupOffset[group];
    return offset + featureId * latentSpaceDimension;
}
void GfmGradientCalc::unittest(){
    Problem data;
    data.l = 1;
    data.y = new int8_t[data.l];
    data.weight = new float[data.l];
    data.x = new FeatureNode * [data.l];
    uint32_t features[] = {0,(1<<29)|1,(2<<29),(3<<29),(3<<29)|1,0xFFFFFFFF};
    void *tmp = reinterpret_cast<void*>(features);
    data.x[0] = reinterpret_cast<FeatureNode*>(tmp);
    data.y[0] = 1;
    Problem::list_problem_struct(data,0);
    GfmGradientCalc gradient(&data);
    Real *mem = new Real[data.l];
    gradient.set_memory(mem);
    Real wholePara[] = {0, 0, 0.15, 0.12, 0.2, -0.1, 0.15, -0.12, 0.3, 0.9, 0.15, 0.12, 0.1, -0.1};
    Real w1[] = {0, 0, 0.15, 0.12, 0.2, -0.1};
    Real w2[] = {0, 0, 0.15, -0.12, 0.3, 0.9};
    Real w3[] = {0, 0, 0.15, 0.12, 0.1, -0.1};
    Real g1[] = {0,0,0,0,0,0}; 
    Real g2[] = {0,0,0,0,0,0}; 
    Real g3[] = {0,0,0,0,0,0}; 
    Real tg1[] = {-0.468765724106, 0, 0.00779813, 0.0062385, -0.17857095, 0.0424089};
    Real tg2[] = {-0.468765724106, 0, -0.20314645, 0.03126275, 0.0031621, 0.00948631};
    Real tg3[] = {-0.468765724106, 0, -0.15626987, 0.10936696, -0.15701455, 0.096075};
    gradient.set_epsilon(1e-4);
    gradient.set_group_lambda(0.01);
    std::vector<uint32_t> groupDimension;
    for(uint32_t i = 0; i < 4; ++i)  groupDimension.push_back(2);
    gradient.init_factor_info(wholePara,2,groupDimension);
    gradient.set_active_group(1);
    double loss = gradient(w1,g1);
    std::cout<<"loss="<<loss<<std::endl;
    assert(std::abs(loss - 0.651466) < 1e-6);
    vec_print(g1,6,std::cout,"g1");
    uint32_t i = 0; 
    for(;i < 6; ++i) assert(std::abs(g1[i] - tg1[i]) < 1e-6);

    gradient.set_active_group(2);
    loss = gradient(w2,g2);
    vec_print(g2,6,std::cout,"g2");
    for(i = 0; i < 6; ++i) assert(std::abs(g2[i] - tg2[i]) < 1e-6);

    gradient.set_active_group(3);
    loss = gradient(w3,g3);
    vec_print(g3,6,std::cout,"g3");
    for(i = 0; i < 6; ++i) assert(std::abs(g3[i] - tg3[i]) < 1e-6);

}

