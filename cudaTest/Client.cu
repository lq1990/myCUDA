#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include "cuBLAS_study.h"


int main()
{
	// getInfoOfDevice();

	// calcSquare();

	// ------------- gpu ----------------
	clock_t t_begin, t_end;
	t_begin = clock();

	// fn
	cublas_demo();

	t_end = clock();
	printf("\n==========================================\ngpu, time needed: %lf s\n",
		(double)(t_end - t_begin) / CLOCKS_PER_SEC);

	// shmem_scan: 0.633s, 0.457s, 0.488s
	// global_scan: 0.508s, 0.455s

	// -------------- cpu -------------------
	/*
	t_begin = clock();

	float in[ARRAY_SIZE];
	float out[ARRAY_SIZE];
	for (int i = 0; i < ARRAY_SIZE; i++)
	{
		in[i] = float(i);
	}

	for (int i = 0; i < ARRAY_SIZE; i++)
	{
		out[i] = in[i] * in[i] * in[i];
	}

	printf("last res: %f\n", out[ARRAY_SIZE - 1]);

	t_end = clock();
	printf("cpu, time needed: %lf s\n", (double)(t_end - t_begin) / CLOCKS_PER_SEC);
*/

	getchar();
	return 0;
}
