#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

__inline__ void modify(cublasHandle_t handle, float * m, int ldm, int n, int p, int q, float alpha, float beta);

int cublas_demo(void);
