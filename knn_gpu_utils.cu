/**
 *
 * Implementations of CUDA kernels and functions used to compute the
 * k nearest neightbors.
 *
 * Design choice: Every block has 128 threads, which is the maximum
 * data dimension. Thus, one block can compute the distance between
 * to vectors with equal or less than 128 dimensions.
 *
 * Author: Christos Nikolaou
 * Date: August 2014
 *
 */


#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

extern "C" {
#include "knn_gpu_utils.h"
}

// Number of threads per block.
#define NUM_THREADS 128

// Define CUDA condition check.
#define CUDA_CHECK(condition) \
/* Code block avoids redefinition of cudaError_t error */ \
do { \
cudaError_t error = condition; \
if (error != cudaSuccess) \
printf("%s\n", cudaGetErrorString(error)); \
} while (0)


// Function to compute the difference between two vectors. 
__global__ void compute_diff(double* X, double* Y, double* diff, int N) {
  
  // Use shared memory for faster computation and reduction
  __shared__ double Z[NUM_THREADS];
  unsigned int tid = threadIdx.x;
  unsigned int i = blockDim.x*blockIdx.x + threadIdx.x;
  
  // Save the difference in the appropriate possition.
  double tmp = 0;

  if (i < N) {
    tmp = X[i] - Y[i];
  }

  Z[tid] = tmp * tmp;

  __syncthreads();

  // Perform reduction
  for (int offset = blockDim.x/2; offset > 0; offset >>=1) {

    if (tid < offset) {
      Z[tid] += Z[tid + offset];
    }
    __syncthreads();
  }

  // Save the sum at the appropriate position in the diff vector.
  if (tid == 0) diff[blockIdx.x] = Z[0];

}

/*
__global__ void compute_sqdiff(double* X, double* Y, double* Z, int N) {

  int index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index < N) {
    double tmp = (X[index] - Y[index]);
    Z[index] = tmp*tmp;
  }

}

__global__ void reduce(double* X, double* sum, int N) {

  __shared__ double sdata[NUM_THREADS];
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
  
  double tmp = 0;

  if (tid < N) {
    tmp = X[tid];
  }
  sdata[tid] = tmp;
  __syncthreads();

  for (int offset = blockDim.x/2; offset > 0; offset >>= 1) {
    if(threadIdx.x < offset) {
      sdata[threadIdx.x] += sdata[threadIdx.x + offset];
    }

    __syncthreads();
  }
  for (unsigned int s = 1; s < blockDim.x; s *= 2) {
    
    if (tid % (2*s) == 0) {
      sdata[tid] += sdata[tid + s];
    }

    __syncthreads();
  }

  if (tid == 0) sum[blockIdx.x] = sdata[0];
}
*/

// Testing function
extern "C"
void say_hello() {
  printf("Hallo!!!\n");
  return;
}

// Print values of GPU arrays
void print_GPU_Mat(double *mat, int length) {

  double *hostmat = (double*)malloc(length*sizeof(double));

  cudaMemcpy(hostmat, mat, length*sizeof(double), cudaMemcpyDeviceToHost);

  for (int i = 0; i < length; ++i) {
    printf("mat[%d] = %f\n", i, hostmat[i]);
  }

  free(hostmat);
}

// Print values of CPU arrays 
void print_CPU_Mat(double *mat, int length) {

  for (int i = 0; i < length; ++i) {
    printf("mat[%d] = %f\n", i, mat[i]);
  }

}


// Compute the euclidean between the N-dimensional vectors X and Y.
extern "C"
double euclidean_distance(double *X, double *Y, int N){

  // Define cuda error
  cudaError_t cudaerr;

  // Variable defined to return the difference
  double retVal;
/*
  printf("--- Print matrix X ---\n");
  print_CPU_Mat(X, N);
  printf("--- Print matrix Y ---\n");
  print_CPU_Mat(Y, N);
*/

  // Define block and grid size
  dim3 blockSize(NUM_THREADS,1);
  int num_blocks = N/NUM_THREADS + 1;
  dim3 gridSize(num_blocks,1);

//  printf("Block size = %d, grid size = %d, number of elements (N) = %d\n", 
//          NUM_THREADS, num_blocks, N);

  // Define device arrays and pass data from the CPU to GPU
  double *A, *B, *RetMat;
  CUDA_CHECK(cudaMalloc((void**) &A, N*sizeof(double)));
  CUDA_CHECK(cudaMalloc((void**) &B, N*sizeof(double)));
  CUDA_CHECK(cudaMalloc((void**) &RetMat, N*sizeof(double)));

  CUDA_CHECK(cudaMemcpy(A, X, N*sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(B, Y, N*sizeof(double), cudaMemcpyHostToDevice));

  /*
  // check errors
  cudaerr = cudaGetLastError();
  if (cudaerr != cudaSuccess)
    printf("Error: %s\n", cudaGetErrorString(cudaerr));

  printf("Computed the squared difference!\n");
 
  printf("--- Print matrix A ---\n");
  print_GPU_Mat(A,N);
  printf("--- Print matrix B ---\n");
  print_GPU_Mat(B,N);
  printf("--- Print matrix RetMat ---\n");
  print_GPU_Mat(RetMat,N);
*/ 

  // Define the array to hold the computed difference between vectors
  double *reduced;
  CUDA_CHECK(cudaMalloc((void**) &reduced, num_blocks*sizeof(double)));

  compute_diff<<<gridSize, blockSize>>>(A,B,reduced,N);

//  reduce<<<gridSize, blockSize>>>(RetMat,reduced,N);

  // Check for kernel errors
  cudaerr = cudaGetLastError();
  if (cudaerr != cudaSuccess)
    printf("Error: %s\n", cudaGetErrorString(cudaerr));


  //printf("--- Print reduced value ---\n");
  //print_GPU_Mat(reduced, 1);
  CUDA_CHECK(cudaMemcpy(&retVal, &reduced[0], sizeof(double), 
              cudaMemcpyDeviceToHost));
  //printf("Reduction completed! Returning value is %f\n", retVal);

  // Free device memory space
  cudaFree(A); cudaFree(B); cudaFree(RetMat); cudaFree(reduced);

  return (double)retVal;

}

