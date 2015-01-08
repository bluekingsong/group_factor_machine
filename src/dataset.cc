#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <cctype>
#include <cerrno>
#include "dataset.h"
#include "log.h"

bool Problem::read_from_binary(std::ifstream& inStream){
    inStream.read((char*)(&l),sizeof(uint64_t));
    std::cout<<"[INF]number of instance="<<l<<std::endl;
    y = new int8_t[l];
    weight = new float[l];
    if(!inStream.read(reinterpret_cast<char*>(y),sizeof(int8_t) * l)){
        std::cerr<<"read y failed.\n";
        return false;
    }
    uint64_t count = 0;
    for(uint64_t i = 0; i < l; ++i){
        count += y[i];
    }
    std::cout<<"[INF]clicks="<<count<<std::endl;
    if(!inStream.read(reinterpret_cast<char*>(weight),sizeof(float) * l)){
        std::cerr<<"read weight failed.\n";
        return false;
    }
    uint8_t *num_features = new uint8_t[l];
    if(!inStream.read(reinterpret_cast<char*>(num_features),sizeof(uint8_t) * l)){
        std::cerr<<"read num_features failed.\n";
        return false;
    }
    uint64_t total_features = 0;
    for(uint64_t i = 0; i < l; ++i){
        total_features += num_features[i];
    }
    std::cout<<"[INF]need total features="<<total_features<<std::endl<<std::endl;
    uint32_t *mem = new uint32_t[total_features];
    if(!inStream.read(reinterpret_cast<char*>(mem),sizeof(uint32_t) * total_features)){
        std::cerr<<"read mem failed. read bytes="<<inStream.gcount()<<"\n";
        return false;
    }
    x = new FeatureNode*[l];
    x[0] = (FeatureNode*)mem;
    for(uint64_t i = 1; i < l; ++i){
        x[i] = x[i - 1] + num_features[i-1];
    }
    delete[] num_features;
    return true;
}
void Problem::shuffle(){
    for(uint64_t i = 1; i < l; ++i){
        uint64_t j = rand() % (l - 1) + 1;
        std::swap(y[i],y[j]);
        std::swap(weight[i],weight[j]);
        std::swap(x[i],x[j]);
    }
}
void Problem::list_problem_struct(const Problem& prob,uint64_t i){
    std::cout<<"number of instances="<<prob.get_instance_num();
    std::cout<<"the "<<i<<"th line of data file is:"<<std::endl;
    for(uint32_t j = 0; j < i; ++j){
        std::cout<<(int32_t)prob.y[j]<<" "<<prob.weight[j];
        FeatureNode *x = prob.x[j];
        while (!x->is_end()){
            std::cout<<" "<<(uint32_t)x->get_group_id()<<":"<<x->get_feature_id();
            x += 1;
        }
        std::cout<<std::endl;
    }
    std::cout<<std::endl;
}

