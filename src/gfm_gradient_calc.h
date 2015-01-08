#ifndef _GFM_GRADIENT_CALC_H
#define _GFM_GRADIENT_CALC_H
#include <iostream>
#include <vector>
#include "dataset.h"
#include "gradient_calc.h"
#include "gflags/gflags.h"

typedef double Real;

DECLARE_bool(pos_weighting);

class GfmGradientCalc : public GradientCalc {
  protected:
    std::vector<uint32_t> groupDimension;
    std::vector<uint64_t> groupOffset;
    uint32_t activeGroup;
    uint32_t latentSpaceDimension;  // latent factor space dimension
    const Real *parameter; // the whole parameter vector, oringal is the matrix,
                           //which each a row is the latent factor for a feature id.
    Real *groupSum; // group[0] is for sum of all group, C1+C2+C3+. ,  Ci=Xi*Wi
    Real *partialSum; // C1+C2, C2+C3,C1+C3
    double epsilon; // for calculate graident and avoid overflow
    double groupLambda;
    double groupLassoLoss;
  public:
    GfmGradientCalc(const Problem* _data,double _intercept = 0):GradientCalc(_data,_intercept){
        epsilon = 1e-8;
        groupLambda = 1;
    }
    virtual AucUti& test_data(const Problem& dataset);
    void set_epsilon(double _epsilon){
        epsilon = _epsilon;
    }
    void set_group_lambda(double lambda){
        groupLambda = lambda;
    }
    double get_group_lasso_loss()const{    return groupLassoLoss;   }
    void init_factor_info(const Real *_parameter,uint32_t latent_factor_dimension,const std::vector<uint32_t>& _groupDimension){
        latentSpaceDimension = latent_factor_dimension;
        parameter = _parameter;
        groupDimension = _groupDimension;
        groupSum = new Real[ latentSpaceDimension * groupDimension.size() ];
        partialSum = new Real[ latentSpaceDimension * groupDimension.size() ];
        groupOffset = GfmGradientCalc::construct_group_offset(groupDimension,latentSpaceDimension);
    }
    bool set_active_group(uint32_t _group){
        if(_group >= groupDimension.size() || _group == 0){
            return false;
        }
        activeGroup = _group;
        return true;
    }
    virtual uint32_t get_parameter_size()const{
        if(0 == activeGroup){
            return groupDimension[0];
        }
        return groupDimension[0] + latentSpaceDimension * groupDimension[activeGroup];
    }
    virtual double operator()(const Real *w, std::vector<uint32_t>& g_index, double& g_val);
    virtual double operator()(const Real *w, std::map<uint32_t,double>& g_dict);
    virtual double operator()(const Real *w, Real * g);
    virtual ~GfmGradientCalc(){
        //GradientCalc::~GradientCalc();
        if(groupSum != 0){
            delete[] groupSum;
        }
        if(partialSum != 0){
            delete[] partialSum;
        }
    }
    static std::vector<uint64_t> construct_group_offset(const std::vector<uint32_t>& groupDimension,uint32_t latentSpaceDimension);
    static uint64_t calc_offset(const std::vector<uint64_t>& groupOffset,uint32_t group,uint32_t featureId,uint32_t latentSpaceDimension);
    static double predict_signal(const FeatureNode *instance,const Real *wholeParameter,const std::vector<uint64_t>& groupOffset,
                                uint32_t activeGroupId, Real *factor_sum,uint32_t k,const Real *w);
    static void unittest();
};
#endif
