#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cublas_v2.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

const int M = 1000; // A.n_rows
const int N = 500; // A.n_cols

#define SIZE 50
#define LOOPS 10000
#define IDX2C(i,j,ld) (((j)*(ld))+(i))
// index to C/C++

/* get thread id: 1D block and 2D grid */
#define get_tid() (blockDim.x * (blockIdx.x+blockIdx.y*gridDim.x)+threadIdx.x)

/* get block id: 2D grid */
#define get_bid() (blockIdx.x+blockIdx.y*gridDim.x)


// time needed: [s]
//			cuda	arma
// 5000:	0.948	2.958
// 10000:	1.182	5.775
// 15000:	1.557	8.742
// 20000:	1.873	11.364
// 25000:	2.219	14.194

void cudaComputeAdd();

void cudaComputeMul();

void cudaFnL2();

/*
	host To device To host.
	y = alpha *A * x + beta * y
	
*/
void h2D2H(float alpha, float* h_in_A, float* h_in_x, float beta, float* h_y);

__global__ void global_elememtMul(float* d_in_vec1, float* d_in_vec2,  int size, float* d_out);

/*
	elementwise mul of vec or mat
*/
void elementMul(float* h_in_vec1, float* h_in_vec2, int size, float* h_res);

void elementMulRun();

/*
	按列方向展示矩阵
*/
void printMat(float* mat, int n_rows, int n_cols);

/*
	一列一列展示vector
*/
void printVec(float* vec, int size);

__global__ void vec_add(float* x, float* y, float* z, int n);

void vec_add_host();

