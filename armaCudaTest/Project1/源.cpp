#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <armadillo>

#include "cudaRun.h"

using namespace std;
using namespace arma;

void armaCompute() {
	clock_t t_begin, t_end;
	t_begin = clock();
	// ====================================

	mat A(SIZE, SIZE);
	// init A with random value
	for (int i = 0; i < SIZE; i++) {
		for (int j = 0; j < SIZE; j++) {
			A(i, j) = 1;
		}
	}

	mat res(SIZE, SIZE); // result
	res = A;
	// A + A + ... + A
	for (int i = 0; i < loops; i++)
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

int main() {
	// 对比 cuda 和 CPU的arma计算速度
	// cuda
	cudaCompute();

	// arma
	armaCompute();

	system("pause");
	return 0;
}