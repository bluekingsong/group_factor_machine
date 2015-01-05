#ifndef _GROUP_FACTOR_MACHINE_H
#define _GROUP_FACTOR_MACHINE_H
#include <vector>
#include "dataset.h"
typedef double Real;
typedef unsigned int uint32_t;

class GroupFactorMachine {
  public:
    GroupFactorMachine(){
        epsilon = 1e-8;
        intercept = 0;
    }
    void set_dataset(const Problem * _data){
        dataset = _data;
    }
    void set_test_dataset(const Problem * _data){
        test_data = _data;
    }
    void set_latent_space_dimension(uint32_t K){
        latentSpaceDimension = K;
    }
    void set_epsilon(double _epsilon){
        epsilon = _epsilon;
    }
    void set_group_dimension(std::vector<uint32_t>& _groupDimension){
        groupDimension = _groupDimension;
    }
    void train(const std::string& optimizerName,uint32_t groupMaxIter);
    void set_intercept(double _intercept){
        intercept = _intercept;
    }
    void init_parameter();
  private:
    double epsilon;
    double intercept;
    const Problem *dataset;
    const Problem *test_data;
    uint32_t latentSpaceDimension;
    std::vector<uint32_t> groupDimension;
    Real *parameter;
};
#endif
