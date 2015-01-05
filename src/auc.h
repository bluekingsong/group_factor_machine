#ifndef SGD_UTILITY_AUC_H__
#define SGD_UTILITY_AUC_H__
#include <stdint.h>

class AucUti {
  public:
    explicit AucUti(double _pctrUpperBound = 0.5,uint32_t _bucketSize = 1000000);
    void add_instance(double pctr,uint32_t label);
    double auc();
    void clear();
    ~AucUti();
  private:
    double pctrUpperBound;
    uint32_t bucketSize;
    uint32_t *impression_cnt_vec;
    uint32_t *click_cnt_vec;
};
#endif
