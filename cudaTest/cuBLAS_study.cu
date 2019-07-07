#include "cuBLAS_study.h"


#define M 3
#define N 2

/*
	把坐标(i,j)转化为cublas数据格式坐标的函数；
	cublas是列优先
	0-based indexing
*/
#define IDX2C(i,j,ld) (((j)*(ld))+(i))

static __inline__ void modify(cublasHandle_t handle, float* devPtrA, int ldm, 
	int n, int p, int q, float alpha, float beta) {
	cublasSscal(handle, n - q, &alpha, &devPtrA[IDX2C(p,q,ldm)], ldm);
	cublasSscal(handle, ldm - p, &beta, &devPtrA[IDX2C(p,q,ldm)], 1);
}

int cublas_demo(void) {
	cudaError_t cudaStat;
	cublasStatus_t stat;
	cublasHandle_t handle;

	int i, j;
	float* devPtrA; // devPtrA on the device
	float* a = 0; // a on the host
	a = (float*)malloc(M*N * sizeof(*a));

	for (j = 0; j < N; j++) {
		for (i = 0; i < M; i++) {
			a[IDX2C(i, j, M)] = (float)(i*M + j + 1);
		}
	}
	/*
		1	7	13	19	25	31
		2	8
		3	9
		4
		5
	*/

	cudaMalloc((void**)&devPtrA, M*N * sizeof(*a));
	cublasCreate(&handle);

	cublasSetMatrix(M, N, sizeof(*a), a, M, devPtrA, M);

	modify(handle, devPtrA, M, N, 1, 2, 16.0f, 12.0f);

	cublasGetMatrix(M, N, sizeof(*a), devPtrA, M, a, M);

	cudaFree(devPtrA);
	cublasDestroy(handle);

	for (j = 0; j < N; j++) {
		for (i = 0; i < M;i++) {
			printf("%7.0f", a[IDX2C(i,j,M)]);
		}
		printf("\n");
	}

	free(a);

	return EXIT_SUCCESS;
}

int fn_level1()
{
	cublasStatus_t stat;
	cublasHandle_t handle;
	const int size = 4;
	float h_in[size] = {
		 -2., 1., 3., 0.
	};
	float h_out[size];
	printVec(h_in, size);

	float* d_in;
	cudaMalloc((void**)&d_in, size * sizeof(float));

	// 此时cublas开始大显身手
	cublasCreate(&handle);

	cublasSetVector(size, sizeof(float), h_in, 1, d_in, 1);

	// 找到vec中最大数绝对值的index, res 是1-based indexing
	int idx= -1;
	cublasIsamax(handle, size, d_in, 1, &idx); 
	printf("index of max mag(val):%d\n",idx);

	// index of min magnitude(val)
	cublasIsamin(handle, size, d_in, 1, &idx);
	printf("index of min mag(val):%d\n",idx);

	// sum of mag(val)
	float sum;
	cublasSasum(handle, size, d_in, 1, &sum);
	printf("sum of mag(val):%f\n", sum);

	// y[i] = alpha * x[i] + y[i]
	float alpha = 2.;
	float h_y[size] = {0., 0., 0., 0.};
	float* d_y;
	cudaMalloc((void**)&d_y, size * sizeof(float));
	cudaMemcpy(d_y, h_y, size * sizeof(float), cudaMemcpyHostToDevice);
	stat = cublasSaxpy(handle, size, &alpha, d_in, 1, d_y, 1);
	printf("stat: %d\n",stat); // 0 means CUBLAS_STATUS_SUCCESS
	cudaMemcpy(h_y, d_y, size * sizeof(float), cudaMemcpyDeviceToHost); // cpy back
	printVec(h_y, size); // -4.0  2.0  6.0  0.0
	
	// copy x into y, copy d_y on the device
	float* h_y2;
	h_y2 = (float*)malloc(size * sizeof(float));
	float* d_y2;
	cudaMalloc((void**)&d_y2, size * sizeof(float));
	cublasScopy(handle, size, d_y, 1, d_y2, 1);
	cudaMemcpy(h_y2, d_y2, size * sizeof(float), cudaMemcpyDeviceToHost);
	printVec(h_y2, size);

	// dot x y, dot(d_in, d_in)
	float res_dot;
	cublasSdot(handle, size, d_in, 1, d_in, 1, &res_dot);
	printf("h_in:\n");
	printVec(h_in, size);
	printf("res dot of h_in: %f\n", res_dot);

	// norm2 of h_in 即平方 求和 再根号
	float res_nrm2_h_in;
	cublasSnrm2(handle, size, d_in, 1, &res_nrm2_h_in);
	printf("res_nrm2_h_in: %f\n", res_nrm2_h_in);

	// scal, x[i] = alpha * x[i] => h_in
	printf("scal 1.5 of h_in: \n");
	float* h_scal;
	h_scal = (float*)malloc(size * sizeof(float));
	alpha = 1.5;
	cudaMemcpy(d_in, h_in, size * sizeof(float), cudaMemcpyHostToDevice);
	cublasSscal(handle, size, &alpha, d_in, 1);
	cudaMemcpy(h_scal, d_in, size * sizeof(float), cudaMemcpyDeviceToHost);
	printVec(h_scal, size);

	// swap x and y, x <=> y
	printf("swap: \n");
	float h_swap1[size] = { 1, 22, 33, 44 };
	float h_swap2[size] = {2, 222, 333, 444};
	float h_out1[size];
	float h_out2[size];
	
	float* d_swap1;
	float* d_swap2;
	cudaMalloc((void**)&d_swap1, size * sizeof(float));
	cudaMalloc((void**)&d_swap2, size * sizeof(float));
	cudaMemcpy(d_swap1, h_swap1, size * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_swap2, h_swap2, size * sizeof(float), cudaMemcpyHostToDevice);
	cublasSswap(handle, size, d_swap1, 1, d_swap2, 1);
	cudaMemcpy(h_out1, d_swap1, size * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_out2, d_swap2, size * sizeof(float), cudaMemcpyDeviceToHost);
	printf("h_out1:\n");
	printVec(h_out1, size);
	printf("h_out2:\n");
	printVec(h_out2, size);


	//cublasGetVector(10, sizeof(float), d_in, 1, h_out, 1);
	// show h_out
	/*for (int i = 0; i < 10; i++)
	{
		printf("%f\n", h_out[i]);
	}*/

	free(h_y2);
	free(h_scal);
	cudaFree(d_in);
	cudaFree(d_y);
	cudaFree(d_y2);
	cublasDestroy(handle);

	return 0;
}

int fn_level2()
{
	cublasStatus_t stat;
	cublasHandle_t handle;
	cublasCreate(&handle);

	// var on the host
	float h_in_mat[M][N];
	float h_in_A[M*N] = {
		0, 1, 2, 3, 4, 5
	};

	/*
		in mem:
			0, 3,
			1, 4,
			2, 5
	*/

	printf("A:\n");
	printMat(h_in_A, 3, 2);

	float* h_in_x;
	float* h_in_y;
	float* h_out_y;
	//h_in_A = (float*)malloc(M*N * sizeof(float));
	h_in_x = (float*)malloc(M * sizeof(float));
	h_in_y = (float*)malloc(N * sizeof(float));
	h_out_y = (float*)malloc(N * sizeof(float));
	float count = 0.0f;
	//for (int i = 0; i < M;i++) {
	//	for (int j = 0; j < N; j++) {
	//		h_in_mat[i][j] = count++;
	//		/*
	//			0 1
	//			2 3
	//			4 5
	//		*/
	//	}
	//}

	// 把h_in_mat行优先存储 转为 h_in_A列优先
	/*for (int i = 0; i < M; i++) {
		for (int j = 0; j < N; j++) {
			h_in_A[IDX2C(i,j,M)] = h_in_mat[i][j];
		}
	}*/

	for (int i = 0; i < M; i++) {
		h_in_x[i] = float(1);
	}
	for (int i = 0; i < N; i++) {
		h_in_y[i] = float(0);
	}

	printf("x:\n");
	printVec(h_in_x, M);

	// var on the device
	float* d_in_A;
	float* d_in_x;
	float* d_in_y;
	cudaMalloc((void**)&d_in_A, M*N*sizeof(float));
	cudaMalloc((void**)&d_in_x, M*sizeof(float));
	cudaMalloc((void**)&d_in_y, N*sizeof(float));
	cudaMemcpy(d_in_A, h_in_A, M*N*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_in_x, h_in_x, M*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_in_y, h_in_y, N*sizeof(float), cudaMemcpyHostToDevice);

	// gemv, y = alpha * A * x + beta * y, => y=A'*x
	float alpha = 1;
	float beta = 0;
	stat = cublasSgemv(handle, CUBLAS_OP_T, 
		M, N,
		&alpha, d_in_A, M, d_in_x, 1, 
		&beta, d_in_y, 1);
	printf("stat: %d\n", stat);

	cudaMemcpy(h_out_y, d_in_y, N * sizeof(float), cudaMemcpyDeviceToHost);
	printf("gemv, y=A'*x:\n");
	printVec(h_out_y, N);


	//free(h_in_A);
	free(h_in_x);
	free(h_in_y);
	free(h_out_y);
	cudaFree(d_in_A);
	cudaFree(d_in_x);
	cudaFree(d_in_y);
	cublasDestroy(handle);
	return 0;
}

int fn_level3()
{
	cublasStatus_t stat;
	cublasHandle_t handle;
	cublasCreate(&handle);

	// var on the host
	const int m = 3;
	const int k = 2;
	const int n = 4;
	float h_in_A[m*k] = {
		0, 2, 4, 1, 3, 5
	};

	/*
		A in mem:
			0, 1,
			2, 3,
			4, 5
	*/

	float h_in_B[k*n] = {
		0, 1, 2, 3, 4, 5, 6, 7
	};

	/*
		B in mem:
			0, 2, 4, 6,
			1, 3, 5, 7
	*/

	float h_in_C[m*n] = {
		0, 0, 0, 0,
		1, 4, 7, 10,
		2, 5, 8, 11
	};

	printf("A:\n");
	printMat(h_in_A, m, k);
	printf("B:\n");
	printMat(h_in_B, k, n);

	// var on the device
	float* d_in_A;
	float* d_in_B;
	float* d_in_C;
	cudaMalloc((void**)&d_in_A, m*k * sizeof(float));
	cudaMalloc((void**)&d_in_B, k*n * sizeof(float));
	cudaMalloc((void**)&d_in_C, m*n * sizeof(float));
	cudaMemcpy(d_in_A, h_in_A, m*k * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_in_B, h_in_B, k*n * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_in_C, h_in_C, m*n * sizeof(float), cudaMemcpyHostToDevice);

	// gemm, C = alpha * A * B + beta * C => C=A*B
	float alpha = 0.1f;
	float beta = 0.0f;

	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
		m, n, k, 
		&alpha, d_in_A, m,
		d_in_B, k, 
		&beta, d_in_C, m);

	float h_out_C[m*n];
	cudaMemcpy(h_out_C, d_in_C, m*n * sizeof(float), cudaMemcpyDeviceToHost);
	printf("C=0.1 * A*B\n");
	printMat(h_out_C, m, n);

	cudaFree(d_in_A);
	cudaFree(d_in_B);
	cudaFree(d_in_C);
	cublasDestroy(handle);
	return 0;
}

int matSetGet()
{
	// var on the host
	float* h_in;
	float* h_out;
	h_in = (float*)malloc(M*N * sizeof(float));
	h_out = (float*)malloc(M*N * sizeof(float)); // h_in, h_out 都要分配空间
	for (int i = 0; i < M*N; i++) {
			h_in[i] = (float)(rand()%10);
	}

	// show h_in
	printf("h_in:\n");
	for (int i = 0; i < M*N; i++) {
		printf("%f\t",h_in[i]);
		if (i%N==(N-1))
		{
			printf("\n");
		}
	}

	// var on the device
	float* d_in;
	//cudaMalloc((void**)&d_in, M * N * sizeof(float));
	cudaMalloc((void**)&d_in, M*N * sizeof(float));


	// cuBLAS
	cublasStatus_t stat;
	cublasHandle_t handle;
	stat = cublasCreate(&handle);
	stat = cublasSetMatrix(M, N, sizeof(float), h_in, M, d_in, M);
	printf("\nset matrix stat:%d\n",stat);

	stat = cublasGetMatrix(M, N, sizeof(float), d_in, M, h_out, M);
	printf("\nget matrix stat:%d\n",stat);

	// show h_out
	printf("\nh_out:\n");
	for (int i = 0; i < M*N; i++) {
		printf("%f\t", h_out[i]);
		if (i%N == (N - 1))
		{
			printf("\n");
		}
	}

	// free
	cudaFree(d_in);
	cublasDestroy(handle);
	free(h_in);
	free(h_out);

	return 0;
}

void printVec(float * vec, int size)
{
	for (int i = 0; i < size;i++) {
		printf("%f\t",vec[i]);
	}
	printf("\n");
}

/*
	以列优先展示mat
*/
void printMat(float* vals, int rows, int cols) {
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			printf("%f\t", vals[j*rows + i]);
		}
		printf("\n");
	}
	printf("\n");
}


void cudaCompute()
{
	clock_t t_begin, t_end;
	t_begin = clock();



	t_end = clock();
	double dt = (double)(t_end - t_begin) / CLOCKS_PER_SEC;
	printf("===================================\n");
	printf("cuda, time needed: %f\n", dt);
}


