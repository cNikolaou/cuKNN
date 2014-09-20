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
 * a grid of blocks with rows (y dimension) equal to the number of queries
 * (max 1000), an the number of columns (x dimension) will be analogus to
 * the number of data points. Because different GPUs tend to have different
 * global memory size, we will compute the distance between Q queries and
 * (at most) 2^16 data points in GPU.
 * 
 * It is easy to change the code a little bit if there are more constraints,
 * or go to a previous version, if there is a larger global memory available
 * on the device.
 *
 *
 * Previous version avialable through git and github:
 * https://github.com/cNikolaou/cuKNN
 *
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
#define MAX_THREADS 128
// Maximum number of blocks available in each dimension
#define MAX_BLOCKS 65535 

// Define the maximum number of points that will be used for the computation
// for each for-loop (in compute_distance_gpu() function). You can change 
// the value based on the available memory on your system.
#define MAX_DATA_POINTS 2


// Define CUDA condition check.
#define CUDA_CHECK(condition) \
/* Code block avoids redefinition of cudaError_t error */ \
do { \
  cudaError_t error = condition; \
  if (error != cudaSuccess) \
  printf("%s\n", cudaGetErrorString(error)); \
} while (0)


// Print values of GPU arrays; used for debugging
void print_gpu_mat(const double *mat, const int length) {

  double *hostmat = (double*)malloc(length*sizeof(double));

  cudaMemcpy(hostmat, mat, length*sizeof(double), cudaMemcpyDeviceToHost);

  for (int i = 0; i < length; ++i) {
    printf("mat[%d] = %f\n", i, hostmat[i]);
  }

  free(hostmat);
}

// Print values of CPU arrays; used for debugging
void print_cpu_mat(const double *mat, const int length) {

  for (int i = 0; i < length; ++i) {
    printf("mat[%d] = %f\n", i, mat[i]);
  }

}

// Print device informations; used for debugging
void printDevProp(cudaDeviceProp devProp) {
    
    printf("Major revision number:         %d\n",  devProp.major);
    printf("Minor revision number:         %d\n",  devProp.minor);
    printf("Name:                          %s\n",  devProp.name);
    printf("Total global memory:           %lu\n",  devProp.totalGlobalMem);
    printf("Total shared memory per block: %lu\n",  devProp.sharedMemPerBlock);
    printf("Total registers per block:     %d\n",  devProp.regsPerBlock);
    printf("Warp size:                     %d\n",  devProp.warpSize);
    printf("Maximum memory pitch:          %lu\n",  devProp.memPitch);
    printf("Maximum threads per block:     %d\n",  devProp.maxThreadsPerBlock);
    for (int i = 0; i < 3; ++i)
        printf("Maximum dimension %d of block:  %d\n", i, devProp.maxThreadsDim[i]);
    for (int i = 0; i < 3; ++i)
        printf("Maximum dimension %d of grid:   %d\n", i, devProp.maxGridSize[i]);
    printf("Clock rate:                    %d\n",  devProp.clockRate);
    printf("Total constant memory:         %lu\n",  devProp.totalConstMem);
    printf("Texture alignment:             %lu\n",  devProp.textureAlignment);
    printf("Concurrent copy and execution: %s\n",  (devProp.deviceOverlap ? "Yes" : "No"));
    printf("Number of multiprocessors:     %d\n",  devProp.multiProcessorCount);
    printf("Kernel execution timeout:      %s\n",  (devProp.kernelExecTimeoutEnabled ?"Yes" : "No"));
    return;

}

// Print information for all the available devices; used for debugging
void print_devices_data() {
    int devCount;
    cudaGetDeviceCount(&devCount);
    printf("CUDA Device Query...\n");
    printf("There are %d CUDA devices.\n", devCount);
 
    for (int i = 0; i < devCount; ++i)
    {
        // Get device properties
        printf("\nCUDA Device #%d\n", i);
        cudaDeviceProp devProp;
        cudaGetDeviceProperties(&devProp, i);
        printDevProp(devProp);
    }  
}

// Function to compute the difference between two vectors. 
__global__ void compute_dist(const double* data, const double* query, 
                             double* dist, const int D, const int N,
                             const int offset) {
  
  // Use shared memory for faster computation and reduction (in each block)
  __shared__ double Z[DIM_THREADS][MAX_THREADS/DIM_THREADS];
  
  // Thread index in block
  unsigned int tix = threadIdx.x;
  unsigned int tiy = threadIdx.y;
  
  // Index in current block for each 'data' matrix dimension
  unsigned int data_ind = threadIdx.x + threadIdx.y*D + 
                          blockIdx.x*blockDim.y*D;

  // Index in current grid line (y-dimesnion) for every query computation                          
  unsigned int query_ind = threadIdx.x + blockIdx.y*D;

/*
  unsigned int i = blockIdx.x*blockDim.x*blockDim.y + 
                   threadIdx.y*blockDim.x + threadIdx.x;
*/

  // Save the difference in the appropriate possition.
  double tmp = 0;

  // When thread's index is less than data's dimenension, then compute distance
  if (tix < D && data_ind < N*D) {
    tmp = data[data_ind] - query[query_ind];
  }

  Z[tix][tiy] = tmp * tmp;

  // Synchronize threads before reduction
  __syncthreads();

  // Perform reduction
  for (int offset = blockDim.x/2; offset > 0; offset >>=1) {

    if (tix < offset) {
      Z[tix][tiy] += Z[tix + offset][tiy];
    }
    __syncthreads();
  }

  // Save the sum at the appropriate position in the diff vector.
  if (tix == 0)// && data_ind < N*D) 
    dist[threadIdx.y + blockIdx.x*blockDim.y + N*blockIdx.y + offset*MAX_DATA_POINTS] = Z[tix][tiy]; //+ N*blockIdx.y

}


// Compute the euclidean between each D-dimensional row of 'data' matrix and 
// the D-dimensional row of 'queries' matrix.
extern "C"
void compute_distance_gpu(const double *data, const double *queries,
                          const int D, const int Q, const int N,
                          double *dist) {

  print_devices_data();

  // Define cuda error
  cudaError_t cudaerr;

  // Define block size
  const int block_y_dim = MAX_THREADS/DIM_THREADS;
  dim3 blockSize(DIM_THREADS,block_y_dim);
 
  // Define grid size
  int num_blocks_x, num_blocks_y;

  // if number of 'data' D-dimensiona arrays are not equally divided by 
  // the number of data points per block (block_y_dim), then grid's
  // x dimension will have a block that doesn't compute the distance
  // between block_y_dim points.
/*  if (N%block_y_dim == 0)
    num_blocks_x = (N/MAX_DATA_POINTS)/block_y_dim;
  else
    num_blocks_x = (N/MAX_DATA_POINTS)/block_y_dim + 1;  
    
  num_blocks_y = Q; //= num_blocks_x/MAX_BLOCKS + 1;
*/
  num_blocks_x = MAX_DATA_POINTS/block_y_dim;
  num_blocks_y = Q;
  dim3 gridSize(num_blocks_x,num_blocks_y);


  printf("Block size = (%d,%d), grid size = (%d,%d), D = %d, Q = %d, N = %d\n", 
          DIM_THREADS, block_y_dim, num_blocks_x, num_blocks_y, D, Q, N);
/**/

  // Define and allocate the device space that will hold the appropriate data
  double *deviceData, *deviceQueries, *deviceDist; 
  
  printf("Maximum memory used during computations: %d\n", (D*MAX_DATA_POINTS+Q*D+Q*N)*sizeof(double));

  printf("Allocating device memory for the data matrix.\n");
  CUDA_CHECK(cudaMalloc((void**) &deviceData, 
                        MAX_DATA_POINTS*D*sizeof(double)));
  printf("Allocating device memory for the queries matrix.\n");
  CUDA_CHECK(cudaMalloc((void**) &deviceQueries, Q*D*sizeof(double)));
  printf("Allocating device memory for the distance matrix.\n");
  CUDA_CHECK(cudaMalloc((void**) &deviceDist, Q*N*sizeof(double)));

//  printf("Transfering 'data' matrix from host to device.\n");
//  CUDA_CHECK(cudaMemcpy(deviceData, data, N*D*sizeof(double), 
//                        cudaMemcpyHostToDevice));
  printf("Transfering 'queries' matrix from host to device.\n");
  CUDA_CHECK(cudaMemcpy(deviceQueries, queries, Q*D*sizeof(double), 
                        cudaMemcpyHostToDevice));

/*  printf("--- All Data Matrix ---\n");
  print_cpu_mat(data, N*D);
  printf("--- Queries Matrix ---\n");
  print_gpu_mat(deviceQueries, Q*D);
*/
  int max_iterations = N/MAX_DATA_POINTS;

  if (N%MAX_DATA_POINTS != 0) {
    max_iterations++;
  }

  printf("Max iterations %d\n", max_iterations);
  int offset;

  for (int iter = 0; iter < max_iterations; ++iter) {
    
//    printf("Transfering 'data' matrix from host to device. Iter = %d\n", iter);
    
    // Offset in dist matrix
    offset = iter * MAX_DATA_POINTS;
    // offset*D equals to the values
    CUDA_CHECK(cudaMemcpy(deviceData, &data[offset*D], 
                          MAX_DATA_POINTS*D*sizeof(double),
                          cudaMemcpyHostToDevice));
    
//    printf("--- Data Matrix ---\n");
//    print_gpu_mat(deviceData, MAX_DATA_POINTS*D);

//    printf("Call kernel for distance computation.\n");
    compute_dist<<<gridSize, blockSize>>>(deviceData,deviceQueries,
                                          deviceDist,D,N,iter);
  }

/*  
  printf("--- Data Matrix ---\n");
  print_gpu_mat(deviceData, N*D);
  printf("--- Queries Matrix ---\n");
  print_gpu_mat(deviceQueries, Q*D);
*/  


  // Check for kernel errors
  cudaerr = cudaGetLastError();
  if (cudaerr != cudaSuccess)
    printf("Error: %s\n", cudaGetErrorString(cudaerr));


  printf("Transfering 'dist' matrix from device to host.\n");
  CUDA_CHECK(cudaMemcpy(dist, deviceDist, Q*N*sizeof(double), 
                        cudaMemcpyDeviceToHost));

  // Only for debugging purposes.

  int i, qi;
  int max_qi = Q;
  // if N is greater than 10K, then print only the first 10K elements
  int max_i = (N > 10000) ? 10000 : N; 
  
  for(qi=0; qi<max_qi; qi++){
    for(i=0; i<max_i; i++){  
      printf("qi = %d, i = %d, dist = %f\n", qi, i, dist[qi*N + i]);
    }
  }
/**/

  cudaFree(deviceData); 
  cudaFree(deviceQueries);
  cudaFree(deviceDist);
}
