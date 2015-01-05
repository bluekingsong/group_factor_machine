#include <cstring>
#include <cmath>
#include <iostream>
#include "vec_op.h"


double vec_dot(const Real *vec1,const Real *vec2,uint32_t vec_len){
	double result = 0;
	for(uint32_t i = 0; i < vec_len; ++i)	result +=  vec1[i] * vec2[i];
	return result;
}

void vec_add(Real *result_vec,const Real *vec1,const Real *vec2,uint32_t vec_len,double factor1,double factor2){
	for(int i=0;i<vec_len;++i){
        Real v1 = vec1[i];
        Real v2 = vec2[i];
        result_vec[i]=factor1*v1+factor2*v2;
    }
}

void vec_cpy(Real *dest,const Real *src,uint32_t vec_len,double srcFactor ){
	if(0 == dest || 0 == src || dest == src)	return;
	if( srcFactor == 1.0 )
		std::memcpy(dest,src,sizeof(Real) * vec_len);
	else{
		for(uint32_t i = 0; i < vec_len; ++i)    dest[i] = srcFactor * src[i];
	}
}

double vec_l1_norm(const Real *vec,uint32_t vec_len){
	double result = 0.0;
	for(int i = 0; i < vec_len; ++i)	result += std::abs(vec[i]);
	return result;
}

void vec_mul(Real *vec,uint32_t vec_len,double factor){
	for(int i = 0;i < vec_len; ++i) vec[i] *= factor;
}

void vec_print(Real *vec,uint32_t vec_len,std::ostream& ostream,const std::string& name){
    ostream<<"Vec["<<name<<"]:";
    for(uint32_t i = 0; i < vec_len; ++i){
        ostream<<vec[i]<<" ";
    }
    ostream<<std::endl;
}
