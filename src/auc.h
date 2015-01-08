#ifndef SGD_UTILITY_AUC_H__
#define SGD_UTILITY_AUC_H__
#include <stdint.h>

class AucUti {
  public:
    explicit AucUti(double _pctrUpperBound = 1.0,uint32_t _bucketSize = 1000000);
    void add_instance(double pctr,uint32_t label);
    double auc();
    void clear();
    double logloss()const { return logloss_; }
    ~AucUti();
    static double LogLoss(double t,int8_t y){
        if(y == 1)    return  t < -25 ? -t : -std::log(1.0 / (1.0 + std::exp(-t)));
        else           return  t > 25  ? t  : -std::log(1.0 - 1.0 / (1.0 + std::exp(-t)));
    }
  private:
    double pctrUpperBound;
    uint32_t bucketSize;
    uint32_t *impression_cnt_vec;
    uint32_t *click_cnt_vec;
    double logloss_;
};
#endif
