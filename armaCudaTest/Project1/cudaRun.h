#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cublas_v2.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define SIZE 500
#define loops 25000

// time needed: [s]
//			cuda	arma
// 5000:	0.948	2.958
// 10000:	1.182	5.775
// 15000:	1.557	8.742
// 20000:	1.873	11.364
// 25000:	2.219	14.194

void cudaCompute();