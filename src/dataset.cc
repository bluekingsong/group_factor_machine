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
    std::cerr<<"l="<<l<<std::endl;
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
    std::cerr<<"read clicks="<<count<<std::endl;
    float first_w;
    //inStream>>first_w;
    //std::cout<<"first_w="<<first_w<<std::endl;
    //return false;
    if(!inStream.read(reinterpret_cast<char*>(weight),sizeof(float) * l)){
        std::cerr<<"read weight failed.\n";
        return false;
    }
    double w_sum = 0;
    for(uint32_t i = 0; i < l; ++i){
      //  inStream>>weight[i];
        w_sum += weight[i];
        if(i < 10) std::cout<<"w["<<i<<"]="<<weight[i]<<std::endl;
    }
    std::cerr<<"read w sum="<<w_sum<<std::endl;
    uint8_t *num_features = new uint8_t[l];
    if(!inStream.read(reinterpret_cast<char*>(num_features),sizeof(uint8_t) * l)){
        std::cerr<<"read num_features failed.\n";
        return false;
    }
    uint64_t total_features = 0;
    for(uint64_t i = 0; i < l; ++i){
        total_features += num_features[i];
    }
    std::cerr<<"need total features="<<total_features<<std::endl;
    std::cerr<<"sizeof FeatureNode = "<<sizeof(FeatureNode)<<" sizeof uint32_t = "<<sizeof(uint32_t)<<std::endl;
    //FeatureNode *mem = new FeatureNode[total_features];
    uint32_t *mem = new uint32_t[total_features];
    if(!inStream.read(reinterpret_cast<char*>(mem),sizeof(uint32_t) * total_features)){
        std::cerr<<"read mem failed. read bytes="<<inStream.gcount()<<"\n";
        return false;
    }
    std::cout<<"mem= "<<mem[0]<<" "<<mem[1]<<" "<<mem[2]<<std::endl;
    x = new FeatureNode*[l];
    x[0] = (FeatureNode*)mem;
    for(uint64_t i = 1; i < l; ++i){
        x[i] = x[i - 1] + num_features[i-1];
    }
    delete[] num_features;
    return true;
}

void Problem::list_problem_struct(const Problem& prob,uint64_t i){
	printf("number of instances=%d\n",prob.l);
	printf("the %dth line of data file is:\n",i);
	FeatureNode *x = prob.x[i];
	while (!x->is_end()){
		printf("%d:%d ",x->get_group_id(),x->get_feature_id());
		x += 1;
	}
	printf("\n");
}

