#ifndef _GROUP_FACTOR_MACHINE_H
#define _GROUP_FACTOR_MACHINE_H
#include <vector>
#include "dataset.h"
#include "gflags/gflags.h"

typedef double Real;
DECLARE_double(epsilon);
DECLARE_double(intercept);
DECLARE_uint64(groupMaxIter);
DECLARE_string(optName);
DECLARE_double(init_stdv);
DECLARE_string(train_file);
DECLARE_string(test_file);
DECLARE_double(gropLambda);
DECLARE_uint64(maxIter);

class GroupFactorMachine {
  public:
    GroupFactorMachine(){
        epsilon = 1e-8;
        intercept = 0;
    }
    void set_dataset(Problem * _data){
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
    void train(const std::string& optimizerName = FLAGS_optName,uint64_t groupMaxIter = FLAGS_groupMaxIter);
    void set_intercept(double _intercept){
        intercept = _intercept;
    }
    void init_parameter();
  private:
    double epsilon;
    double intercept;
    Problem *dataset;
    const Problem *test_data;
    uint32_t latentSpaceDimension;
    std::vector<uint32_t> groupDimension;
    Real *parameter;
};
#endif
