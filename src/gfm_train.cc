#include <fstream>
#include <vector>
#include "gfm_gradient_calc.h"
#include "optimizer.h"
#include "lbfgs.h"
#include "group_factor_machine.h"
#include "dataset.h"
#include "gflags/gflags.h"

int main(int argc,char **argv){
    google::ParseCommandLineFlags(&argc, &argv, true);
    std::vector<google::CommandLineFlagInfo> flags;
    google::GetAllFlags(&flags);
    for(uint32_t i = 0; i < flags.size(); ++i){
        std::cout<<google::DescribeOneFlag(flags[i])<<std::endl;
    }
    Problem dataset;
    std::ifstream fin(FLAGS_train_file.c_str(),std::ios_base::in|std::ios_base::binary);
    std::ifstream test_fin(FLAGS_test_file.c_str(),std::ios_base::in|std::ios_base::binary);
    if(!fin.is_open() || !test_fin.is_open()){
        std::cerr<<"[ERR]open file "<<argv[1]<<" or "<<argv[2]<<" failed."<<std::endl;
        return 1;
    }
    dataset.read_from_binary(fin);
    Problem test_data;
    test_data.read_from_binary(test_fin);
    fin.close();
    test_fin.close();
    GroupFactorMachine model;
    model.set_dataset(&dataset);
    model.set_test_dataset(&test_data);
    model.set_latent_space_dimension(10);
    std::vector<uint32_t> groupDimension;
    //0:30 1:658688 2:4260788 3:525189
    groupDimension.push_back(30);
    groupDimension.push_back(658688);
    groupDimension.push_back(4260788);
    groupDimension.push_back(525189);
    model.set_group_dimension(groupDimension);
    model.train();
    return 0;
}

