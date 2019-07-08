#include "cudaRun.h"

#define IDX2C(i,j,ld) (((j)*(ld))+(i))

void cudaComputeAdd()
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
	// 本质上A B都是一维的，所以可以使用level 1 中的指令
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
	// C = alpha * A * B + beta * C 会把计算结果放到 C 中
	float alpha = 1.0f;
	float beta = 1.0f;

	for (int i = 0; i < LOOPS; i++)
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

void cudaComputeMul()
{
	clock_t t_begin, t_end;
	t_begin = clock();
	// ====================================




	// ====================================
	t_end = clock();
	printf("---------------------------\n");
	printf("cuda, time needed: %f s\n",
		(double)(t_end - t_begin) / CLOCKS_PER_SEC);
	printf("============================\n");
}

void cudaFnL2()
{
	clock_t t_begin, t_end;
	t_begin = clock();
	// ====================================
	cublasStatus_t stat;
	cublasHandle_t handle;
	cublasCreate(&handle);

	// 探究 level 1 2 3 的常用方法
	// ----------- level 1 --------------------
	



	// ----------- level 2 --------------------
	// gemv: y = alpha * A' * x + beta * y
	float alpha = 1, beta = 0; // y=A' * x
	
	// var on the host
	float* h_in_A;
	float* h_in_x;
	float* h_y;
	h_in_A = (float*)malloc(M*N * sizeof(float));
	h_in_x = (float*)malloc(M * sizeof(float));
	h_y = (float*)malloc(N*sizeof(float));
	int count = 0;
	for (int i = 0; i < M; i++)
	{
		for (int j = 0; j < N; j++)
		{
			h_in_A[IDX2C(i, j, M)] = count++;
		}
	}
	/*
		0, 1, 2,
		3, 4, 5,
		.
		.
		.
	*/
	/*printf("A:\n");
	printMat(h_in_A, M, N);*/

	for (int i = 0; i < M; i++)
	{
		h_in_x[i] = 1.0f;
	}
	/*printf("x:\n");
	printVec(h_in_x, M);*/

	// var on the device
	float* d_in_A;
	float* d_in_x;
	float* d_y;
	cudaMalloc((void**)&d_in_A, M*N * sizeof(float));
	cudaMalloc((void**)&d_in_x, M * sizeof(float));
	cudaMalloc((void**)&d_y, N * sizeof(float));
	cudaMemcpy(d_in_A, h_in_A, M*N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_in_x, h_in_x, M * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, h_y, N * sizeof(float), cudaMemcpyHostToDevice);

	// compute with cuBLAS
	for (int i = 0; i < LOOPS; i++)
	{
		cublasSgemv(handle,
			CUBLAS_OP_T, // A'
			M, N, // n_rows, n_cols of A
			&alpha,
			d_in_A, M, // M 是dla，即leading dimension of A
			d_in_x, 1,
			&beta,
			d_y, 1); // y stores the result
	}

	cudaMemcpy(h_y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);
	/*printf("y:\n");
	printVec(h_y, N);*/

	// free
	free(h_in_A);
	free(h_in_x);
	free(h_y);
	cudaFree(d_in_A);
	cudaFree(d_in_x);
	cudaFree(d_y);

	// ----------- level 3 --------------------


	cublasDestroy(handle);

	// ====================================
	t_end = clock();
	printf("\n");
	printf("cuda, time needed: %f s\n",
		(double)(t_end - t_begin) / CLOCKS_PER_SEC);
	printf("=====================\n");
}

void h2D2H(float alpha, float * h_in_A, float * h_in_x, float beta, float * h_y)
{

}

__global__ void global_elememtMul(float * d_in_vec1, float * d_in_vec2, int size, float * d_out)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < size)
	{
		int v1 = d_in_vec1[tid];
		int v2 = d_in_vec2[tid];
		d_out[tid] = v1 * v2;
	}

}

void elementMul(float * h_in_vec1, float * h_in_vec2, int size, float * h_res)
{
	// var on the host are already init and assigned

	// var on the device
	float* d_in_vec1;
	float* d_in_vec2;
	float* d_out;
	cudaMalloc((void**)&d_in_vec1, size * sizeof(float));
	cudaMalloc((void**)&d_in_vec2, size * sizeof(float));
	cudaMalloc((void**)&d_out, size * sizeof(float));
	cudaMemcpy(d_in_vec1, h_in_vec1, size * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_in_vec2, h_in_vec2, size * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_out, h_res, size * sizeof(float), cudaMemcpyHostToDevice);
	
	
	// launch kernel
	global_elememtMul << <2, 1024 >> > (d_in_vec1, d_in_vec2, size, d_out);

	// copy back to host
	cudaMemcpy(h_res, d_out, size * sizeof(float), cudaMemcpyDeviceToHost);

}

void elementMulRun()
{
	// var on the host
	float* h_in_vec1;
	float* h_in_vec2;
	int size = 1025;
	float* h_res;
	h_in_vec1 = (float*)malloc(size * sizeof(float));
	h_in_vec2 = (float*)malloc(size * sizeof(float));
	h_res = (float*)malloc(size * sizeof(float));

	for (int i = 0; i < size; i++)
	{
		h_in_vec1[i] = float(i);
		h_in_vec2[i] = float(1);
	}
	printf("vec1: \n");
	printVec(h_in_vec1, size);
	printf("vec2: \n");
	printVec(h_in_vec2, size);

	// ----------------- cuda fn --------------------
	elementMul(h_in_vec1, h_in_vec2, size, h_res);

	printf("elem Mul: \n");
	printVec(h_res, size);

}



void printMat(float * mat, int n_rows, int n_cols)
{
	for (int i = 0; i < n_rows; i++)
	{
		for (int j = 0; j < n_cols; j++)
		{
			printf("%f\t", mat[j*n_rows + i]);
		}
		printf("\n");
	}
}

void printVec(float * vec, int size)
{
	for (int i = 0; i < size; i++)
	{
		printf("%f\t", vec[i]);
	}
	printf("\n");
}

__global__ void vec_add(float * x, float * y, float * z, int n)
{
	int tid = get_tid();

	if (tid < n)
	{
		z[tid] += x[tid] + y[tid];
	}
}

void vec_add_host() {
	int N = 1000000;
	int nbytes = N * sizeof(float);

	/* 1D block */
	int bs = 256;

	/* 2D grid */
	int s = ceil(sqrt((N + bs - 1.) / bs));
	dim3 grid = dim3(s, s);

	float *dx = NULL, *hx = NULL;
	float *dy = NULL, *hy = NULL;
	float *dz = NULL, *hz = NULL;

	int itr = 30; // loops
	int i;
	float th, td;

	/* allocate GPU mem */
	cudaMalloc((void**)&dx, nbytes);
	cudaMalloc((void**)&dy, nbytes);
	cudaMalloc((void**)&dz, nbytes);

	/* allocate CPU mem */
	hx = (float*)malloc(nbytes);
	hy = (float*)malloc(nbytes);
	hz = (float*)malloc(nbytes);

	/* init */
	for (int i = 0; i < N; i++)
	{
		hx[i] = 1.;
		hy[i] = 1.;
		hz[i] = 1.;
	}

	/* copy data to GPU */
	cudaMemcpy(dx, hx, nbytes, cudaMemcpyHostToDevice);
	cudaMemcpy(dy, hy, nbytes, cudaMemcpyHostToDevice);
	cudaMemcpy(dz, hz, nbytes, cudaMemcpyHostToDevice);

	/* call GPU */
	cudaThreadSynchronize();
	for (int i = 0; i < itr; i++)
	{
		// launch kernel on the device
		vec_add << <grid, bs >> > (dx, dy, dz, N);
	}
	cudaThreadSynchronize(); 
	// 由于kernel调用对host而言是异步的，所以使用此行 wait for device to finish
	cudaMemcpy(hz, dz, nbytes, cudaMemcpyDeviceToHost);
	for (int i = 0; i < 10; i++)
	{
		printf("%f\n", hz[i]);
	}

	cudaFree(dx);
	cudaFree(dy);
	cudaFree(dz);
	free(hx);
	free(hy);
	free(hz);

}





