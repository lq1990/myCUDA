#include "cudaRun.h"

#define IDX2C(i,j,ld) (((j)*(ld))+(i))

void cudaCompute()
{
	clock_t t_begin, t_end;
	t_begin = clock();
	// ====================================

	cublasStatus_t stat;
	cublasHandle_t handle;
	cublasCreate(&handle);
	// var on the host
	float* h_in; // 存储新的结果
	float* h_in_A; // 被加数，不变
	float* h_out;
	float* h_in_B;

	h_in = (float*)malloc(SIZE*SIZE * sizeof(float));
	h_in_A = (float*)malloc(SIZE*SIZE * sizeof(float));
	h_out = (float*)malloc(SIZE*SIZE * sizeof(float));
	h_in_B = (float*)malloc(SIZE*SIZE * sizeof(float));
	for (int i = 0; i < SIZE*SIZE; i++)
	{
		h_in[i] = (float)(1);
		h_in_A[i] = (float)(1);
	}

	// B 的主对角线上是1， 其余是0
	for (int i = 0; i < SIZE; i++)
	{
		for (int j = 0; j < SIZE; j++)
		{
			if (i==j)
			{
				h_in_B[IDX2C(i,j,SIZE)] = (float)(1);
			}
			else
			{
				h_in_B[IDX2C(i,j,SIZE)] = (float)(0);
			}
		}
	}

	// var on the device
	float* d_in;
	float* d_in_A;
	float* d_in_B;
	cudaMalloc((void**)&d_in, SIZE*SIZE * sizeof(float));
	cudaMalloc((void**)&d_in_A, SIZE*SIZE * sizeof(float));
	cudaMalloc((void**)&d_in_B, SIZE*SIZE * sizeof(float));
	cudaMemcpy(d_in, h_in, SIZE*SIZE * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_in_A, h_in_A, SIZE*SIZE * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_in_B, h_in_B, SIZE*SIZE * sizeof(float), cudaMemcpyHostToDevice);
	// cublas level 3
	// C = alpha * A * B + beta * C
	float alpha = 1.0f;
	float beta = 1.0f;

	for (int i = 0; i < loops; i++)
	{
		cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
			SIZE, SIZE, SIZE,
			&alpha,
			d_in_A, SIZE,
			d_in_B, SIZE,
			&beta,
			d_in, SIZE);
	}

	cudaMemcpy(h_out, d_in, SIZE*SIZE * sizeof(float), cudaMemcpyDeviceToHost);

	// 展示前10个数
	for (int i = 0; i < 10; i++)
	{
		printf("%f\n",h_out[i]);
	}

	// free
	free(h_in);
	free(h_out);
	free(h_in_B);
	cudaFree(d_in);
	cudaFree(d_in_B);
	cublasDestroy(handle);
	// ====================================
	t_end = clock();
	printf("---------------------------\n");
	printf("cuda, time needed: %f s\n", 
		(double)(t_end-t_begin)/CLOCKS_PER_SEC);
	printf("============================\n");
}
