#include "cuBLAS_study.h"

#define M 6
#define N 5

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