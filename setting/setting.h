#ifndef __SETTING_H
#define __SETTING_H
#include <assert.h>

enum DType{
	DTypeNONE    = -1,
	DTypeFLOAT32 = 0,
	DTypeDOUBLE  = 1,
};

#define use_dtype 0



#if use_dtype == 0 
#define Real float
#define USE_FLOAT
#elif use_dtype == 1 
#define Real double
#define USE_DOUBLE
#else
#pragma error
#endif

static int GetDType(){
	if (sizeof(Real) == 8) return DTypeDOUBLE;
	if (sizeof(Real) == 4) return DTypeFLOAT32;
	assert(0);
	return DTypeNONE;
}

#ifndef SHOW_DTYPE
#define SHOW_DTYPE  printf("%s : %s : %d, using dtype=%s\n", __FILE__, __FUNCTION__, __LINE__, sizeof(Real)==8?"double":"float32");
#endif

#endif // !__SETTING_H

