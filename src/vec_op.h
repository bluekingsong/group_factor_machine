#ifndef _VECOP_H_
#define _VECOP_H_
#include <istream>
#include <string>

typedef double Real;

// @vec_len the length of vectors
// @retval inner product of vec1 and vec2
double vec_dot(const Real *vec1,const Real *vec2,uint32_t vec_len);

// @result_vec result vector of factor1*vec1+factor2*vec2;
// @vec_len the legnth of vectors
void vec_add(Real *result_vec,const Real *vec1,const Real *vec2, uint32_t vec_len,double factor1,double factor2);

// @dest destination of copy vector @srcFactor * @src
// @vec_len the length of vectors
void vec_cpy(Real *dest,const Real *src,uint32_t vec_len,double srcFactor = 1);

// #retval L1-norm of @vec, i.e. sum of |vec[i]|
double vec_l1_norm(const Real *vec,uint32_t vec_len);

// vec=factor*vec,@factor is a scalar
void vec_mul(Real *vec,uint32_t vec_len,double factor);

void vec_print(Real *vec,uint32_t vec_len,std::ostream& ostream,const std::string& name);
#endif
