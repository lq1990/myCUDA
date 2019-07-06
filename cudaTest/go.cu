#include "cuda_runtime.h"
#include "cublas_v2.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>


const int ARRAY_SIZE = 100;
const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);

void getInfoOfDevice()
{
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);

	int dev;
	for (dev = 0; dev < deviceCount; dev++)
	{
		int driver_version(0), runtime_version(0);
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, dev);
		if (dev == 0)
			if (deviceProp.minor = 9999 && deviceProp.major == 9999)
				printf("\n");
		printf("\nDevice%d:\"%s\"\n", dev, deviceProp.name);
		cudaDriverGetVersion(&driver_version);
		printf("CUDA驱动版本:                                   %d.%d\n", driver_version / 1000, (driver_version % 1000) / 10);
		cudaRuntimeGetVersion(&runtime_version);
		printf("CUDA运行时版本:                                 %d.%d\n", runtime_version / 1000, (runtime_version % 1000) / 10);
		printf("设备计算能力:                                   %d.%d\n", deviceProp.major, deviceProp.minor);
		printf("Total amount of Global Memory:                  %u bytes\n", deviceProp.totalGlobalMem);
		printf("Number of SMs:                                  %d\n", deviceProp.multiProcessorCount);
		printf("Total amount of Constant Memory:                %u bytes\n", deviceProp.totalConstMem);
		printf("Total amount of Shared Memory per block:        %u bytes\n", deviceProp.sharedMemPerBlock);
		printf("Total number of registers available per block:  %d\n", deviceProp.regsPerBlock);
		printf("Warp size:                                      %d\n", deviceProp.warpSize);
		printf("Maximum number of threads per SM:               %d\n", deviceProp.maxThreadsPerMultiProcessor);
		printf("Maximum number of threads per block:            %d\n", deviceProp.maxThreadsPerBlock);
		printf("Maximum size of each dimension of a block:      %d x %d x %d\n", deviceProp.maxThreadsDim[0],
			deviceProp.maxThreadsDim[1],
			deviceProp.maxThreadsDim[2]);
		printf("Maximum size of each dimension of a grid:       %d x %d x %d\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
		printf("Maximum memory pitch:                           %u bytes\n", deviceProp.memPitch);
		printf("Texture alignmemt:                              %u bytes\n", deviceProp.texturePitchAlignment);
		printf("Clock rate:                                     %.2f GHz\n", deviceProp.clockRate * 1e-6f);
		printf("Memory Clock rate:                              %.0f MHz\n", deviceProp.memoryClockRate * 1e-3f);
		printf("Memory Bus Width:                               %d-bit\n", deviceProp.memoryBusWidth);
	}
}

/*
	Kernel
	__global__ 表示：这个function将运行在GPU上。
*/
__global__ void square(float* d_out, float* d_in) {
	int idx = threadIdx.x;
	float f = d_in[idx];
	d_out[idx] = f * f;
}

__global__ void cube(float* d_out, float* d_in) {
	int index = threadIdx.x;
	float f = d_in[index];
	d_out[index] = f * f * f;
}

void calcSquare() {
	const int ARRAY_SIZE = 10;
	const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);

	// generate the input array on the host
	float h_in[ARRAY_SIZE];
	for (int i = 0; i < ARRAY_SIZE; i++) {
		h_in[i] = float(i);
	}
	float h_out[ARRAY_SIZE]; // h_out contains result that is copied from d_out

	// declare GPU memory pointers
	float* d_in;
	float* d_out;

	// allocate GPU memory
	cudaMalloc((void**)&d_in, ARRAY_BYTES);
	cudaMalloc((void**)&d_out, ARRAY_BYTES);

	// transfer the array to GPU
	cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice);

	// launch the kernel on GPU
	square << <1, ARRAY_SIZE >> > (d_out, d_in);
	// <<<加载多少线程块，一个块多少线程>>>

	// copy back the result array to the GPU
	cudaMemcpy(h_out, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost);

	// print out the resulting array
	for (int i = 0; i < ARRAY_SIZE; i++) {
		printf("%f", h_out[i]);
		printf(((i % 4) != 3) ? "\t" : "\n");
	}

	// free GPU memory allocation
	cudaFree(d_in);
	cudaFree(d_out);
}

void calcCube() {

	// cpu, var init
	float h_in[ARRAY_SIZE];
	float h_out[ARRAY_SIZE];
	for (int i = 0; i < ARRAY_SIZE; i++)
	{
		h_in[i] = float(i);
	}

	float* d_in;
	float* d_out;

	// cudaMalloc
	cudaMalloc((void**)&d_in, ARRAY_BYTES);
	cudaMalloc((void**)&d_out, ARRAY_BYTES);

	// cudaMemcpy host to device
	cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice);

	cube << <1, ARRAY_SIZE >> > (d_out, d_in);

	// copy back, device to host
	cudaMemcpy(h_out, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost);

	// show h_out
	/*for (int i = 0; i < ARRAY_SIZE; i++)
	{
		printf("%f\n",h_out[i]);
	}*/
	printf("last res: %f\n", h_out[ARRAY_SIZE-1]);

	// free res on GPU
	cudaFree(d_in);
	cudaFree(d_out);

}

/*
	in global mem
*/
__global__ void global_reduce_kernel(float* d_out, float* d_in) {
	int myId = threadIdx.x + blockDim.x * blockIdx.x; // 区分所有线程块 中每个线程
	int tid = threadIdx.x; // 当前线程块的线程位置
	
	// do reduction in global mem
	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (tid<s) {
			d_in[myId] += d_in[myId + s];
		}
		__syncthreads(); // make sure all adds at one stage are done!
	}
	// 对for的理解：可以把循环拆成 从上到下 一串串的计算，s以折半的速度减少到1就停止了

	// only thread 0 writes result for this block back to global mem
	if (tid==0)
	{
		d_out[blockIdx.x] = d_in[myId];
	}
}

/*
	in shared mem
*/
__global__ void shmem_reduce_kernel(float* d_out, float* d_in) {
	// sdata is allocated in the kernel call: 3rd arg to <<<b,t,shmem>>>
	extern __shared__ float sdata[];

	int myId = threadIdx.x + blockDim.x * blockIdx.x;
	int tid = threadIdx.x;

	// loat shared mem from global mem
	sdata[tid] = d_in[myId];
	__syncthreads(); // make sure entire block is loaded!

	// do reduction in shared mem
	for (unsigned int s = blockDim.x/2; s >0 ; s>>=1)
	{
		if (tid<s)
		{
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}

	// only thread 0 writes result for this block back to global mem
	if (tid == 0)
	{
		d_out[blockIdx.x] = sdata[0];
	}

}

void reduce(float* d_out, float* d_intermediate, float* d_in, int size, bool usesSharedMem) {
	// assume that size is not greater than maxThreadsPerBlock ^ 2
	// and that size is a multiple of maxThreadsPerBlock
	const int maxThreadsPerBlock = 1024;
	int threads = maxThreadsPerBlock;
	int blocks = size / maxThreadsPerBlock;
	if (usesSharedMem)
	{
		shmem_reduce_kernel << <blocks, threads, threads * sizeof(float) >> > (d_intermediate, d_in);
		//<<<块数, 单个块中的线程数, 共享内存大小>>>
	}
	else 
	{
		global_reduce_kernel << <blocks, threads >> > (d_intermediate, d_in);
	}

	// now we're down to one block left, so reduce it
	threads = blocks;
	blocks = 1;
	if (usesSharedMem)
	{

	}
	else
	{
		global_reduce_kernel << <blocks, threads >> > (d_out, d_intermediate);
	}
	
}

/*
	d_out, d_in 是来自 global mem中，所以访问速度慢
*/
__global__ void global_scan(float* d_out, float* d_in) {
	int idx = threadIdx.x; // 多线程下，区分线程，以分工

	float out = 0.00f;
	d_out[idx] = d_in[idx]; // init val of d_out
	__syncthreads(); // init d_out over

	for (int inter = 1; inter <= sizeof(d_in); inter *= 2)
	{
		if (idx-inter>=0)
		{
			out = d_out[idx] + d_out[idx - inter];
			// use local temp var "out" to store result
			// 每个线程都有自己的局部变量out，计算线程自己负责的部分
		}
		__syncthreads();
		if (idx-inter>=0)
		{
			d_out[idx] = out; // 每个线程将自己负责的部分计算结果存储到 应该的位置。
			out = 0.00f;
		}
	}

}

/*
	使用 共享变量暂存来自 global mem的数据
*/
__global__ void shmem_scan(float* d_out, float* d_in) {
	extern __shared__ float sdata[];

	int idx = threadIdx.x;

	float out = 0.0f;
	// init, assign values to sdata[]
	sdata[idx] = d_in[idx];
	__syncthreads();

	for (int inter = 1; inter <= sizeof(d_in); inter *= 2) {
		if (idx - inter >= 0)
		{
			out = sdata[idx] + sdata[idx - inter];
		}
		__syncthreads();

		if (idx-inter >= 0)
		{
			sdata[idx] = out;
			out = 0.0f;
		}

	}

	d_out[idx] = sdata[idx];

}

void scan() {
	const int size = 11;
	const int bytes = size * sizeof(float);

	float h_in[size];
	for (int i = 0; i < size; i++)
	{
		h_in[i] = float(i);
	}
	float h_out[size];

	printf("sizeof(h_in): %d\n",sizeof(h_in));

	// show h_in
	for (int i = 0; i < size; i++)
	{
		printf("%f\t", h_in[i]);
	}
	printf("\n");

	float* d_in;
	float* d_out;

	cudaMalloc((void**)&d_in, bytes);
	cudaMalloc((void**)&d_out, bytes);

	cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

	// launch kernel
	global_scan << <1, size >> > (d_out, d_in);
	//shmem_scan << <1, size, size * sizeof(float)>> > (d_out, d_in); // <<<,,共享变量的内存空间大小>>>

	cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

	// show h_out
	for (int i = 0; i < size; i++)
	{
		printf("%f\t", h_out[i]);
	}
	printf("\n");

	cudaFree(d_in);
	cudaFree(d_out);

}

/*
	d_in 中有128个数，由于只有8个线程处理，
	先把128个数分成16部分,
	一个线程负责16个数。

	d_out_bins维度 3行8列，不同线程负责不同的位置。但二维数组不易构建，使用一维数组代替
	d_out_bins 1行24列

	  thread    1 2 3 4 5 6 7 8
	bin1:		o o o o o o o o
	bin2:		o o o o o o o o
	bin3:		o o o o o o o o

*/
__global__ void global_histo(float* d_out_bins, float* d_in) {
	int tid = threadIdx.x;

	// 每个单线程都要负责 16 个数。比如0到15号位的数属于 第0号线程
	for (int i = 0; i < 128 / 8; i++) {
		int val = d_in[tid*8 + i];
		int pos = val % 3;
		atomicAdd(&(d_out_bins[tid + 8 * pos]), 1);

		/*if (pos==0)
		{
			atomicAdd(&(d_out_bins[tid + 8*0]), 1);
		}
		else if (pos==1)
		{
			atomicAdd(&(d_out_bins[tid + 8*1]), 1);
		}
		else if (pos == 2)
		{
			atomicAdd(&(d_out_bins[tid + 8*2]), 1);
		}*/
	}
}

/*
	d_in 把3行n列（其实是1行3n列）
	d_out 转成 3行1列（亦1行3列）
*/
__global__ void global_bins_reduction(float* d_out, float* d_in) {
	int size = 8;

	int tid = threadIdx.x;

	for (int s = size/2; s > 0; s >>= 1) {
		if (tid < s)
		{
			d_in[tid + 8*0] += d_in[tid + 8*0 + s];
			d_in[tid + 8*1] += d_in[tid + 8*1 + s];
			d_in[tid + 8*2] += d_in[tid + 8*2 + s];
		}
		__syncthreads();
	}

	if (tid == 0)
	{
		d_out[0] = d_in[tid + 8 * 0];
		d_out[1] = d_in[tid + 8 * 1];
		d_out[2] = d_in[tid + 8 * 2];
	}

}

/*
	直方图。
	128个数字，8个线程，3个分组

*/
void histo() {
	const int SIZE = 128;

	// var on host
	float h_in[SIZE];
	float h_out_bins[24];
	for (int i = 0; i < SIZE; i++)
	{
		h_in[i] = float(rand());
		printf("%f\n",h_in[i]);
	}

	// var on device
	float* d_in;
	float* d_out_bins;

	cudaMalloc((void**)&d_in, SIZE*sizeof(float));
	cudaMalloc((void**)&d_out_bins, 24*sizeof(float));

	// cudaMemcpy
	cudaMemcpy(d_in, h_in, SIZE * sizeof(float), cudaMemcpyHostToDevice);

	// launch kernel
	int blocks = 1;
	int threads = 8;
	global_histo << <blocks, threads >> > (d_out_bins, d_in);

	// cudaMemcpy
	cudaMemcpy(h_out_bins, d_out_bins, 24*sizeof(float), cudaMemcpyDeviceToHost);

	printf("sizeof(h_in): %d\n", sizeof(h_in)); // 512 = 128*4
	printf("sizeof(d_out_bins): %d\n", sizeof(d_out_bins)); // 8
	printf("sizeof(*d_out_bins): %d\n", sizeof(*d_out_bins)); // 4
	// 把h_out_bins 转化
	float* d_out;
	cudaMalloc((void**)&d_out, 3 * sizeof(float));
	global_bins_reduction << <1, 4 >> > (d_out, d_out_bins); // kernel

	float h_out[3];
	cudaMemcpy(h_out, d_out, 3 * sizeof(float), cudaMemcpyDeviceToHost);

	/*float h_out[3] = { 0, 0, 0 };
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 8; j++)
		{
			h_out[i] += h_out_bins[8*i+j];
		}
	}*/

	// show h_out_bins
	for (int i = 0; i < 24; i++)
	{
		printf("%f\t", h_out_bins[i]);
		if (i % 8 == 7)
		{
			printf("\n");
		}
	}

	printf("\n");
	// show h_out
	for (int i = 0; i < 3; i++)
	{
		printf("%f\t", h_out[i]);
	}
	printf("\n");

	// free
	cudaFree(d_in);
	cudaFree(d_out_bins);
	cudaFree(d_out);
}


