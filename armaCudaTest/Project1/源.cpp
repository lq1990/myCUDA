#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <armadillo>

#include "cudaRun.h"

using namespace std;
using namespace arma;

/* 
	注意： .cu .cpp 不同的文件类型，右键文件（不是项目）设置 常规-项类型 是不同的
	.cpp默认即可。
	.cu必须手动设置 项类型为 "CUDA C/C++"
	否则程序编译出错
*/

void armaComputeAdd() {
	clock_t t_begin, t_end;
	t_begin = clock();
	// ====================================

	mat A(SIZE, SIZE);
	// init A with random value
	for (int i = 0; i < SIZE; i++) {
		for (int j = 0; j < SIZE; j++) {
			A(i, j) = 0.1;
		}
	}

	mat res(SIZE, SIZE); // result
	res = A;
	// A + A + ... A
	for (int i = 0; i < LOOPS; i++)
	{
		res += A;
	}

	// 展示前10个结果
	for (int i = 0; i < 10; i++)
	{
		printf("%f\n", res(i, 0));
	}

	// ====================================
	t_end = clock();
	printf("----------------------------\n");
	printf("arma, time needed: %f s\n",
		(double)(t_end - t_begin) / CLOCKS_PER_SEC);
	printf("============================\n");
}

void armaComputeMv() {
	clock_t t_begin, t_end;
	t_begin = clock();
	// ====================================
	// y = A*x
	mat A(M, N);
	mat x(M, 1);
	mat y(N, 1);
	int count = 0;
	for (int i = 0; i < M; i++)
	{
		for (int j = 0; j < N; j++)
		{
			A(i, j) = count++;
		}
	}
	//A.print("arma A:");

	for (int i = 0; i < M; i++)
	{
		x(i, 0) = 1;
	}
	//x.print("arma x:");

	// compute y = A' * x
	for (int i = 0; i < LOOPS; i++)
	{
		y = A.t() * x;
	}
	//y.print("arma y:");


	// ====================================
	t_end = clock();
	printf("\n");
	printf("arma, time needed: %f s\n",
		(double)(t_end - t_begin) / CLOCKS_PER_SEC);
	printf("=====================\n");
}

int main() {
	clock_t t_begin, t_end;
	t_begin = clock();
	// ====================================

	// 对比 cuda 和 CPU的arma计算速度
	// cuda
	//cudaComputeMul();
	// arma
	//armaCompute();


	// =======================================
	// cuda
	vec_add_host();

	//elementMulRun();

	// vs arma
	//armaComputeMv();




	// ====================================
	t_end = clock();
	printf("----------------------------\n");
	printf("main, time needed: %f s\n",
		(double)(t_end - t_begin) / CLOCKS_PER_SEC);

	system("pause");
	return 0;
}