/**
 *
 * Implementations of CUDA kernels and functions used to compute the
 * k nearest neightbors.
 *
 * Problem hightlits: Maximum N = 2^20 data points with maximum 
 * dimension 128. Up to Q = 1000 queries with k = 1:8 neightbors
 * for each query.
 *
 * Design choice: Every block has 128 threads, which is the maximum
 * data dimension. Thus, one block can compute the distance between
 * two vectors with equal or less than 128 dimensions. We will start
 * multiple blocks to compute the distance between one data point
 * and all queries. We will start queries->secondary_dim blocks.
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
#define MAX_THREADS 512

// Define CUDA condition check.
#define CUDA_CHECK(condition) \
/* Code block avoids redefinition of cudaError_t error */ \
do { \
cudaError_t error = condition; \
if (error != cudaSuccess) \
printf("%s\n", cudaGetErrorString(error)); \
} while (0)


// Function to compute the difference between two vectors. 
__global__ void compute_diff(double* X, double* Y, double* diff, int D) {
  
  // Use shared memory for faster computation and reduction
  __shared__ double Z[NUM_THREADS][MAX_THREADS/NUM_THREADS];
  unsigned int tix = threadIdx.x;
  unsigned int tiy = threadIdx.y;
  unsigned int i = blockIdx.x*blockDim.x*blockDim.y + threadIdx.y*blockDim.x +
                    threadIdx.x;
  
  // Save the difference in the appropriate possition.
  double tmp = 0;

  if (tix < D) {
    tmp = X[i] - Y[i];
  }

  Z[tix][tiy] = tmp * tmp;

  __syncthreads();

  // Perform reduction
  for (int offset = blockDim.x/2; offset > 0; offset >>=1) {

    if (tix < offset) {
      Z[tix][tiy] += Z[tix + offset][tiy];
    }
    __syncthreads();
  }

  // Save the sum at the appropriate position in the diff vector.
  if (tix == 0) diff[blockIdx.x*blockDim.y + threadIdx.y] = Z[0][tiy];

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
void euclidean_distance(double *X, double *Y, int D, int Q, int N,
                        int index, double *diff) {

  // Define cuda error
  cudaError_t cudaerr;

/*
  printf("--- Print matrix X ---\n");
  print_CPU_Mat(X, D);
  printf("--- Print matrix Y ---\n");
  print_CPU_Mat(Y, D);
*/

  // Define block and grid size
  const int y_dim = MAX_THREADS/NUM_THREADS;
  dim3 blockSize(NUM_THREADS,y_dim);
  const int num_blocks = Q/NUM_THREADS + 1;
  dim3 gridSize(num_blocks,1);
/*
  printf("Block size = %d, grid size = %d, number of elements (N) = %d\n", 
          NUM_THREADS, num_blocks, D);
*/
  // Matrix to prepare the data to pass to device
  double *prepA, *prepB;
  prepA = (double*)malloc(NUM_THREADS*Q*sizeof(double));
  prepB = (double*)malloc(NUM_THREADS*Q*sizeof(double));

  for (int i = 0; i < Q; i++) {
    for (int j = 0; j < NUM_THREADS; j++) {
      if (j < D) {
        prepA[i*NUM_THREADS + j] = X[j];
        prepB[i*NUM_THREADS + j] = Y[i*D + j];
      } else {
        prepA[i*NUM_THREADS + j] = 0;
        prepB[i*NUM_THREADS + j] = 0;
      }
    }
  }
/*
  printf("--- Print matrix prepA ---\n");
  print_CPU_Mat(prepA, Q*NUM_THREADS);
  printf("--- Print matrix prepB ---\n");
  print_CPU_Mat(prepB, Q*NUM_THREADS);
*/

  // Define device arrays and pass data from the CPU to GPU
  double *A, *B, *RetMat;
  CUDA_CHECK(cudaMalloc((void**) &A, Q*NUM_THREADS*sizeof(double)));
  CUDA_CHECK(cudaMalloc((void**) &B, Q*NUM_THREADS*sizeof(double)));
  CUDA_CHECK(cudaMalloc((void**) &RetMat, Q*NUM_THREADS*sizeof(double)));

  CUDA_CHECK(cudaMemcpy(A, prepA, Q*NUM_THREADS*sizeof(double), 
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(B, prepB, Q*NUM_THREADS*sizeof(double), 
                        cudaMemcpyHostToDevice));

/*
  // check errors
  cudaerr = cudaGetLastError();
  if (cudaerr != cudaSuccess)
    printf("Error: %s\n", cudaGetErrorString(cudaerr));

  printf("Computed the squared difference!\n");
 */
/*
  printf("--- Print matrix A ---\n");
  print_GPU_Mat(A,Q*NUM_THREADS);
  printf("--- Print matrix B ---\n");
  print_GPU_Mat(B,Q*NUM_THREADS);
*//*
  printf("--- Print matrix RetMat ---\n");
  print_GPU_Mat(RetMat,D);
*/ 

  // Define the array to hold the computed difference between vectors
  double *reduced;
  CUDA_CHECK(cudaMalloc((void**) &reduced, Q*sizeof(double)));

  compute_diff<<<gridSize, blockSize>>>(A,B,reduced,D);

//  reduce<<<gridSize, blockSize>>>(RetMat,reduced,D);

  // Check for kernel errors
  cudaerr = cudaGetLastError();
  if (cudaerr != cudaSuccess)
    printf("Error: %s\n", cudaGetErrorString(cudaerr));


  double retVal[Q];
  //printf("--- Print reduced value ---\n");
  //print_GPU_Mat(reduced, Q);
  CUDA_CHECK(cudaMemcpy(&retVal[0], &reduced[0], Q*sizeof(double), 
              cudaMemcpyDeviceToHost));
  //printf("Reduction completed! Returning value is %f\n", retVal);

  // Free device memory space
  cudaFree(A); cudaFree(B); cudaFree(RetMat); cudaFree(reduced);

  for (int i = 0; i < Q; i++) {
    diff[index + i*N] = retVal[i];
  }

}

