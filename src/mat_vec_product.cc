#include "mat_vec_product.h"

class HilbertProduct :public MatVecProduct{
  public:
    virtual void operator()(const Real *x,const uint32_t n,Real *y)const{
        for(uint32_t i = 0; i < n; ++i){
            y[i] = 0.0;
            for(uint32_t j = 0; j < n; ++j)    y[i] += x[i] / ( i + j +1 );
        }
    }
};
