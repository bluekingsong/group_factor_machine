#include <cstdlib>
#include <cmath>
#include <cstring>
#include <iostream>
#include "auc.h"

double AucUti::auc(){
    double auc_temp=0.0,no_click=0.0,click_sum=0.0,no_click_sum=0.0,old_click_sum=0.0;
    for(uint32_t i = 0; i < bucketSize; ++i){
        if(0 == impression_cnt_vec[i])    continue;  // no impression in this bucket
        uint32_t num_impression = impression_cnt_vec[i];
        uint32_t num_click = click_cnt_vec[i];
        auc_temp += (click_sum+old_click_sum) * no_click / 2.0;
        old_click_sum = click_sum;
        no_click = 0.0;
        no_click += num_impression-num_click;
        no_click_sum +=num_impression-num_click;
        click_sum += num_click;
    }
    auc_temp += (click_sum+old_click_sum) * no_click / 2.0;
    //std::cout<<"click_sum="<<click_sum<<" imp_sum="<<no_click_sum<<std::endl;
    double auc = auc_temp / (click_sum * no_click_sum);
    return auc;
}
void AucUti::add_instance(double t,uint32_t label){
    logloss_ += AucUti::LogLoss(t,label);
    double pctr = 1.0 / (1.0 + std::exp(-t));
    if(pctr > pctrUpperBound)    pctr = pctrUpperBound;
    uint32_t index = static_cast<uint32_t>((1.0 - pctr / pctrUpperBound) * (bucketSize - 1));
    if(label)    ++click_cnt_vec[index];
    ++impression_cnt_vec[index];
}
AucUti::AucUti(double _pctrUpperBound,uint32_t _bucketSize){
    pctrUpperBound = _pctrUpperBound;
    bucketSize = _bucketSize;
    click_cnt_vec = new uint32_t[bucketSize * 2];
    impression_cnt_vec = click_cnt_vec + bucketSize;
    std::memset(click_cnt_vec,0,sizeof(uint32_t) * bucketSize * 2);
    logloss_ = 0.0;
}
void AucUti::clear(){
    logloss_ = 0.0;
    std::memset(click_cnt_vec,0,sizeof(uint32_t) * bucketSize * 2);
}
AucUti::~AucUti(){
    if(click_cnt_vec)    delete[] click_cnt_vec;
}
