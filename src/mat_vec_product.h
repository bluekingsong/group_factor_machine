#ifndef _MAT_VEC_PRODUCT_H
#define _MAT_VEC_PRODUCT_H

typedef double Real;
typedef unsigned int uint32_t;

class MatVecProduct{
  public:
    // description of A*x=y
    virtual void operator()(const Real *x,const uint32_t n,Real *y)const=0;
};

#endif
