#include <cstdlib>
#include <iostream>
#include <fstream>
#include "dataset.h"
#include "gfm_gradient_calc.h"


int main(int argc,char **argv){
    GfmGradientCalc::unittest();
    return 0;
    if(argc != 2){
        std::cerr<<"usage:"<<argv[0]<<" binary_file"<<std::endl;
        return 1;
    }
    //Problem dataset = read_problem("feature_id.smp");
    const char* filename = argv[1];
    std::ifstream fin(filename,std::ios_base::in|std::ios_base::binary);
    if(!fin.is_open()){
        std::cerr<<"[ERR]open file "<<filename<<" is failed."<<std::endl;
        return 1;
    }
    Problem dataset;
    if(!dataset.read_from_binary(fin)){
        std::cerr<<"[ERR] read failed."<<std::endl;
        return 1;
    }
    Problem::list_problem_struct(dataset,0);
    dataset.free_memory();
    return 0;
}


