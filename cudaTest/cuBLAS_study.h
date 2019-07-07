#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <time.h>

__inline__ void modify(cublasHandle_t handle, float * m, int ldm, int n, int p, int q, float alpha, float beta);

int cublas_demo(void);

/*
	scalar and vector op
*/
int fn_level1();

/*
	matrix-vector op
*/
int fn_level2();


/*
	matrix-matrix op
*/
int fn_level3();

int matSetGet();

void printVec(float* vec, int size);

void printMat(float * vals, int rows, int cols);

void cudaCompute();
