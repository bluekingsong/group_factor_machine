#include "gfm_gradient_calc.h"
#include "optimizer.h"
#include "lbfgs.h"
#include "group_factor_machine.h"
#include "dataset.h"

int main(int argc,char **argv){
    if(argc != 5){
        std::cerr<<argv[0]<<" train_data_file test_data_file optAlgo[gd/bfgs/sbfgs] groupMaxIter"<<std::endl;
        return 1;
    }
    Problem dataset;
    std::ifstream fin(argv[1],std::ios_base::in|std::ios_base::binary);
    std::ifstream test_fin(argv[2],std::ios_base::in|std::ios_base::binary);
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
    model.train(std::string(argv[3]),std::atoi(argv[4]));
    return 0;
}

