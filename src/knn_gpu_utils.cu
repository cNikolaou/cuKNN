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
#define DIM_THREADS 128
#define MAX_THREADS 256
#define MAX_BLOCKS 65535 

// Define CUDA condition check.
#define CUDA_CHECK(condition) \
/* Code block avoids redefinition of cudaError_t error */ \
do { \
cudaError_t error = condition; \
if (error != cudaSuccess) \
printf("%s\n", cudaGetErrorString(error)); \
} while (0)


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


// Function to compute the difference between two vectors. 
__global__ void compute_dist(double* data, double* query, double* dist, 
                              int D, int N) {
  
  // Use shared memory for faster computation and reduction
  __shared__ double Z[DIM_THREADS][MAX_THREADS/DIM_THREADS];
  unsigned int tix = threadIdx.x;
  unsigned int tiy = threadIdx.y;
  
  unsigned int data_ind = threadIdx.x + threadIdx.y*D + blockIdx.x*blockDim.y*D;
  unsigned int query_ind = threadIdx.x + blockIdx.y*D;

/*  unsigned int i = blockIdx.x*blockDim.x*blockDim.y + threadIdx.y*blockDim.x +
                    threadIdx.x;
*/

  // Save the difference in the appropriate possition.
  double tmp = 0;

  if (tix < D && data_ind < N*D) {
    tmp = data[data_ind] - query[query_ind];
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
  if (tix == 0 && data_ind < N*D) 
    dist[threadIdx.y + blockIdx.x*blockDim.y + N*blockIdx.y] = Z[tix][tiy];

}


// Compute the euclidean between the N-dimensional vectors X and Y.
// Function that transfers the data to device memory
extern "C"
void compute_distance_gpu(double *data, double *queries, int D, int Q, int N,
                          double *dist) {

  // Define cuda error
  cudaError_t cudaerr;

  // Define block and grid size
  const int y_dim = MAX_THREADS/DIM_THREADS;
  dim3 blockSize(DIM_THREADS,y_dim);
  int num_blocks_x, num_blocks_y;

  if (N%y_dim == 0)
    num_blocks_x = N/y_dim;
  else
    num_blocks_x = N/y_dim + 1;  
    
  num_blocks_y = Q; //= num_blocks_x/MAX_BLOCKS + 1;

  dim3 gridSize(num_blocks_x,num_blocks_y);
/*
  printf("Block size = (%d,%d), grid size = (%d,%d), D = %d, Q = %d, N = %d\n", 
          DIM_THREADS, y_dim, num_blocks_x, num_blocks_y, D, Q, N);
*/

  // Define and allocate the device space that will hold the appropriate data
  double *deviceData, *deviceQueries, *deviceDist; 
  
  CUDA_CHECK(cudaMalloc((void**) &deviceData, N*D*sizeof(double)));
  CUDA_CHECK(cudaMalloc((void**) &deviceQueries, Q*D*sizeof(double)));
  CUDA_CHECK(cudaMalloc((void**) &deviceDist, Q*N*sizeof(double)));

  CUDA_CHECK(cudaMemcpy(deviceData, data, N*D*sizeof(double), 
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(deviceQueries, queries, Q*D*sizeof(double), 
                        cudaMemcpyHostToDevice));


/*  
  printf("--- Data Matrix ---\n");
  print_GPU_Mat(deviceData, N*D);
  printf("--- Queries Matrix ---\n");
  print_GPU_Mat(deviceQueries, Q*D);
*/  



  compute_dist<<<gridSize, blockSize>>>(deviceData,deviceQueries,
                                          deviceDist,D,N);

  // Check for kernel errors
  cudaerr = cudaGetLastError();
  if (cudaerr != cudaSuccess)
    printf("Error: %s\n", cudaGetErrorString(cudaerr));


  CUDA_CHECK(cudaMemcpy(dist, deviceDist, Q*N*sizeof(double), 
                        cudaMemcpyDeviceToHost));
/*
  int i, qi;

  for(qi=0; qi<Q; qi++){
    for(i=0; i<N; i++){  
      printf("qi = %d, i = %d, dist = %f\n", qi, i, dist[qi*N + i]);
    }
  }
*/

  cudaFree(deviceData); 
  cudaFree(deviceQueries);
  cudaFree(deviceDist);
}
