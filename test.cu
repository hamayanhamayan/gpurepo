#include <stdio.h>
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>

//#define NUM_BLOCKS 4
#define NUM_THREADS 512

#define DATA_NUM 512

__global__ void kernel(int* gdata)
{
	__shared__ int sdata[DATA_NUM];

	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	sdata[tid] = gdata[tid];
	__syncthreads();

	sdata[tid] *= sdata[0];
}

int main(int argc, char **argv)
{
	int* sdata = (int*)malloc(sizeof(int) * DATA_NUM);
	for(int i = 0;i < DATA_NUM;i++) sdata[i] = i + 1;

	int* gdata = NULL;
	checkCudaErrors( cudaMalloc((void**)&gdata, sizeof(int) * DATA_NUM) );
	checkCudaErrors( cudaMemcpy(gdata, sdata, sizeof(int) * DATA_NUM, cudaMemcpyHostToDevice) );

	kernel<<<DATA_NUM/NUM_THREADS, NUM_THREADS>>>(gdata);

	checkCudaErrors( cudaMemcpy(sdata, gdata, sizeof(int) * DATA_NUM, cudaMemcpyDeviceToHost) );

	printf("1～%dの総和→%d\n", DATA_NUM, sdata[1]);

	cudaFree(gdata);
	free(sdata);
	
	cudaDeviceReset();
}
