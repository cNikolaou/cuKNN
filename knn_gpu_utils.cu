/**
 *
 * Implementations of CUDA kernels and functions used to compute the
 * k nearest neightbors.
 *
 * Problem hightlights: (Choosen when to program is called)
 * Maximum N = 2^20 data points with maximum dimension 128. 
 * Up to Q = 1000 queries with k = 1 to 8 neightbors for each query.
 *
 * Design choice: Every block has 128 threads, which is the maximum
 * data dimension. Thus, one block can compute the distance between
 * two vectors with equal or less than 128 dimensions. We will start
 * a grid of blocks with rows (y dimension) equal to the number of queries
 * (max 1000), an the number of columns (x dimension) will be analogus to
 * the number of data points. Because different GPUs tend to have different
 * global memory size, we will compute the distance between Q queries and
 * (at most) 2^16 data points in GPU (defined with the MAX_DATA_POINT 
 * global variable).
 * 
 * It is easy to change the code a little bit if there are more constraints,
 * or go to a previous version, if there is a larger global memory available
 * on the device. In general, it will be easy to fine-tune this code on any
 * GPU.
 *
 *
 * Previous versions avialable from github:
 * https://github.com/cNikolaou/cuKNN
 *
 *
 * Author: Christos Nikolaou
 * Date: August 2014
 *
 */


#include <stdio.h>
#include <float.h>

#include <cuda.h>
#include <cuda_runtime.h>

extern "C" {
#include "knn_gpu_utils.h"
}

// NOTE: DIM_THREADS and MAX_THREADS should be choosen based on GPU's
// architecture.
// Number of threads per data point
#define DIM_THREADS 128
// Number of threads per block. Should be a multiple of DIM_THREADS, so that
// more than one data points will be computed from within a block.
#define MAX_THREADS 128
// Maximum number of blocks available in each dimension (GPU restriction)
#define MAX_BLOCKS 65535 

// Define the maximum number of points that will be used for the computation
// for each for-loop (in compute_distance_gpu() function). You can change 
// the value based on the available memory on your system. For each additional 
// data point you will need additional D*sizeof(double) memory space to sace
// the data point plus Q*D*sizeof(double) memory to save the computed distance.
#define MAX_DATA_POINTS 2

// Define the maximum number of queries that will be used in each loop of the
// for loop, when finding the minimum distances in selection_gpu() function.
// For each additional query there is a need for (approximately) 
// N*sizeof(double) additional memory space needed in the GPU.
#define MAX_QUERIES 1

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
                             double* dist, const int D, const int N) {
  
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
    dist[threadIdx.y + blockIdx.x*blockDim.y+ MAX_DATA_POINTS*blockIdx.y] 
          = Z[tix][tiy];

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
  // There will be Q rows and there will be computed 
  num_blocks_x = MAX_DATA_POINTS/block_y_dim;
  num_blocks_y = Q;
  dim3 gridSize(num_blocks_x,num_blocks_y);

  printf("Parameters when computing distance:\n");
  printf("Block size = (%d,%d), grid size = (%d,%d), D = %d, Q = %d, N = %d\n", 
          DIM_THREADS, block_y_dim, num_blocks_x, num_blocks_y, D, Q, N);
/**/

  // Define and allocate the device space that will hold the appropriate data
  double *deviceData, *deviceQueries, *deviceDist; 
  
//  printf("Maximum memory used during computations: %d\n", (D*MAX_DATA_POINTS+Q*D+Q*N)*sizeof(double));

//  printf("Allocating device memory for the data matrix.\n");
  CUDA_CHECK(cudaMalloc((void**) &deviceData, 
                        MAX_DATA_POINTS*D*sizeof(double))); 
//  printf("Allocating device memory for the queries matrix.\n");
  CUDA_CHECK(cudaMalloc((void**) &deviceQueries, Q*D*sizeof(double)));
//  printf("Allocating device memory for the distance matrix.\n");
  CUDA_CHECK(cudaMalloc((void**) &deviceDist, Q*MAX_DATA_POINTS*sizeof(double)));

  double *tempDist = (double*) malloc(Q*MAX_DATA_POINTS*sizeof(double));
//  printf("Transfering 'data' matrix from host to device.\n");
//  CUDA_CHECK(cudaMemcpy(deviceData, data, N*D*sizeof(double), 
//                        cudaMemcpyHostToDevice));
//  printf("Transfering 'queries' matrix from host to device.\n");
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

//  printf("Max iterations %d\n", max_iterations);
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
                                          deviceDist,D,N);
    // Check for kernel errors
    cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess)
      printf("Error: %s\n", cudaGetErrorString(cudaerr));


//    printf("Transfering 'dist' matrix from device to host.\n");
    CUDA_CHECK(cudaMemcpy(tempDist, deviceDist, Q*MAX_DATA_POINTS*sizeof(double), 
                        cudaMemcpyDeviceToHost));

/*    printf("--- Temp dist ---\n");
    print_gpu_mat(deviceDist, Q*MAX_DATA_POINTS);
    printf("--- Transfer temp dist ---\n");
*/
    for (int qi = 0; qi < Q; qi++) {
        for (int i = 0; i < MAX_DATA_POINTS; i++) {
  //        printf("tempDist[%d] = %f\n", 
  //                  i + qi*MAX_DATA_POINTS, tempDist[i + qi*MAX_DATA_POINTS]);
          dist[i + qi*N + iter*MAX_DATA_POINTS] = 
                                              tempDist[i + qi*MAX_DATA_POINTS];  
        } 
      } 
  }

/*  
  printf("--- Data Matrix ---\n");
  print_gpu_mat(deviceData, N*D);
  printf("--- Queries Matrix ---\n");
  print_gpu_mat(deviceQueries, Q*D);
*/  


  // Only for debugging purposes.
/*
  int max_qi = Q;
  // if N is greater than 10K, then print only the first 10K elements
  int max_i = (N > 10000) ? 10000 : N; 
  
  for (int qi = 0; qi < max_qi; qi++) {
    for (int i = 0; i < max_i; i++) {  
      printf("qi = %d, i = %d, dist = %f\n", qi, i, dist[qi*N + i]);
    }
  }
*/

  cudaFree(deviceData); 
  cudaFree(deviceQueries);
  cudaFree(deviceDist);
}


/* ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ */
/* ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ */
/* ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ */
/* ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ */

__global__ void block_min(const double* dist, const int* index, const int N,
                          double* block_min_dist, int* block_min_indx) {
  
  unsigned int tix = threadIdx.x;
  unsigned int idx = threadIdx.x + blockIdx.x*blockDim.x;
  unsigned int bix = blockIdx.x;

  // Use shared memory for faster computation and reduction (in each block)
  __shared__ double local_min_dist[MAX_THREADS];
  __shared__ double local_min_indx[MAX_THREADS];


  if (idx < N) {
    local_min_dist[tix] = dist[idx];
    local_min_indx[tix] = index[idx];
  } else {
    local_min_dist[tix] = DBL_MAX;
    local_min_indx[tix] = -1;
  }

  __syncthreads();


  for (int offset = blockDim.x/2; offset > 0; offset >>=1) {
    
    if (tix < offset) {

      if (local_min_dist[tix] > local_min_dist[tix + offset]) {
        local_min_dist[tix] = local_min_dist[tix + offset];
        local_min_indx[tix] = local_min_indx[tix + offset];
      }
    }
    __syncthreads();
  }

  // all blocks write their outcome
  if (tix == 0) {
    block_min_dist[bix] = local_min_dist[0];
    block_min_indx[bix] = local_min_indx[0];
  }
}

__global__ void total_min(const double* block_min_dist, 
                          const int* block_min_idx, 
                          const int n_blocks, 
                          int k, double* min_dist, int* min_idx) {

  unsigned int tix = threadIdx.x;
  
  __shared__ double local[MAX_THREADS];

  local[tix] = block_min_dist[tix];

  for (int offset = blockDim.x/2; offset > 0; offset >>=1) {
        
    if (tix < offset) {
      if (local[tix] > local[tix + offset])
        local[tix] = local[tix+offset];  
    }
    __syncthreads();
    
  }

  if (tix == 0) {
    min_dist[k] = block_min_dist[1];  
    min_idx[k] = block_min_dist[0];
  }
  
}

void compute_min() {
  
}

extern "C"
void selection_gpu(double* dist, double* NNdist, 
                   int* NNidx, int N, int Q, int k) {

  // ----- First phase reduction block and grid -----
  // GPU's block dimension
  int block_y_dim = 1;
  dim3 blockSize(MAX_THREADS, block_y_dim);

  // GPU's grid dimension; there will be MAX_QUERIES processed at the same time
  int num_blocks_x = N/MAX_THREADS + 1;
  int num_blocks_y = MAX_QUERIES;
  dim3 gridSize(num_blocks_x, num_blocks_y);

  // ----- Second phase reduction block and grid -----
  int block_y_dim_2 = 1;
  dim3 blockSize_2(MAX_THREADS, block_y_dim_2);

  // GPU's grid dimension; there will be MAX_QUERIES processed at the same time
  int num_blocks_x_2 = num_blocks_x/MAX_THREADS + 1;
  int num_blocks_y_2 = MAX_QUERIES;
  dim3 gridSize_2(num_blocks_x_2, num_blocks_y_2);

  // ----- Third phase reduction block and grid -----
  int block_y_dim_3 = 1;
  dim3 blockSize_3(MAX_THREADS, block_y_dim_3);

  // GPU's grid dimension; there will be MAX_QUERIES processed at the same time
  int num_blocks_x_3 = num_blocks_x_2/MAX_THREADS + 1;
  int num_blocks_y_3 = MAX_QUERIES;
  dim3 gridSize_3(num_blocks_x_3, num_blocks_y_3);


  printf("Parameters when selecting the k neighbors:\n");
  printf("Phase one: block size = (%d,%d), grid size = (%d,%d)\n", 
          MAX_THREADS, block_y_dim, num_blocks_x, num_blocks_y);
  printf("Phase two: block size = (%d,%d), grid size = (%d,%d)\n", 
          MAX_THREADS, block_y_dim_2, num_blocks_x_2, num_blocks_y_2);
  printf("Phase three: block size = (%d,%d), grid size = (%d,%d)\n", 
          MAX_THREADS, block_y_dim_3, num_blocks_x_3, num_blocks_y_3);
  
  if (num_blocks_x_3 > 1) {
    printf("ERROR; you need another reduction phase! Too many blocks!\n");  
  }

/**/

  // Array that holds the index. Used to find NNidx in the kernel.
  int idx[N];

  for (int i = 0; i < N; i++) {
    idx[i] = i;  
  }

  // CPU matrices that will be used to take 
  double *minDist = (double*) malloc(k*sizeof(double));
  int *minIdx = (int*) malloc(k*sizeof(int));

  // GPU arrays that will hold the data
  double *deviceDist, *deviceMinDist;
  int *deviceIdx, *deviceMinIdx;
  
  CUDA_CHECK(cudaMalloc((void**) &deviceDist, MAX_QUERIES*N*sizeof(double)));
  CUDA_CHECK(cudaMalloc((void**) &deviceMinDist, MAX_QUERIES*k*sizeof(double)));
  CUDA_CHECK(cudaMalloc((void**) &deviceIdx, N*sizeof(int)));
  CUDA_CHECK(cudaMalloc((void**) &deviceMinIdx, MAX_QUERIES*k*sizeof(int)));

  // Transfer indices in GPU
  CUDA_CHECK(cudaMemcpy(deviceIdx, idx, N*sizeof(int), 
                        cudaMemcpyHostToDevice));
/*
  // Print 
  printf("--- CPU distance matrix ---\n");
  print_cpu_mat(dist, Q*N);
*/

  double *deviceTmpMinDist, *deviceTmpMinDist_2;
  int *deviceTmpMinIdx, *deviceTmpMinIdx_2;
  
  CUDA_CHECK(cudaMalloc((void**) &deviceTmpMinDist, 
                        MAX_QUERIES*num_blocks_x*sizeof(double)));
  CUDA_CHECK(cudaMalloc((void**) &deviceTmpMinIdx, 
                        MAX_QUERIES*num_blocks_x*sizeof(int)));

  CUDA_CHECK(cudaMalloc((void**) &deviceTmpMinDist_2, 
                        MAX_QUERIES*num_blocks_x_2*sizeof(double)));
  CUDA_CHECK(cudaMalloc((void**) &deviceTmpMinIdx_2, 
                        MAX_QUERIES*num_blocks_x_2*sizeof(int)));


  int max_iterations = Q/MAX_QUERIES;

  for (int iter = 0; iter < max_iterations; ++iter) {
    
    for (int neighbor = 0; neighbor < k; ++neighbor) {
      CUDA_CHECK(cudaMemcpy(deviceDist, &dist[iter*N*MAX_QUERIES], 
                            N*MAX_QUERIES*sizeof(double),
                            cudaMemcpyHostToDevice));
/*    
      printf("--- GPU distance matrix ---\n");
      print_gpu_mat(deviceDist, N*MAX_QUERIES);
  */  
      block_min<<<gridSize, blockSize>>>(deviceDist, deviceIdx, N, 
                                           deviceTmpMinDist, deviceTmpMinIdx);

      block_min<<<gridSize_2, blockSize_2>>>(deviceTmpMinDist, 
                                             deviceTmpMinIdx, num_blocks_x, 
                                             deviceTmpMinDist_2, 
                                             deviceTmpMinIdx_2);

      block_min<<<gridSize_3, blockSize_3>>>(deviceTmpMinDist_2, 
                                             deviceTmpMinIdx_2, num_blocks_x_2, 
                                             deviceMinDist, deviceMinIdx);
/*
      printf("--- GPU minimum distance matrix ---\n");
      print_gpu_mat(deviceMinDist, MAX_QUERIES);
*/
      CUDA_CHECK(cudaMemcpy(minDist, deviceMinDist, 
                            MAX_QUERIES*k*sizeof(double),
                            cudaMemcpyDeviceToHost));    
        
      CUDA_CHECK(cudaMemcpy(minIdx, deviceMinIdx, 
                            MAX_QUERIES*k*sizeof(int),
                            cudaMemcpyDeviceToHost));
/*
      printf("--- GPU minimum index matrix ---\n");
      printf("Index = %d\n", minIdx[0]);
*/
      NNdist[iter*k + neighbor] = minDist[0];
      NNidx[iter*k + neighbor] = minIdx[0];
      dist[iter*N*MAX_QUERIES+minIdx[0]] = 500000;//DBL_MAX;
    }
  }

  cudaFree(deviceDist);
  cudaFree(deviceMinDist);
  cudaFree(deviceIdx);
  cudaFree(deviceMinIdx);
  cudaFree(deviceTmpMinDist);
  cudaFree(deviceTmpMinIdx);
  cudaFree(deviceTmpMinDist_2);
  cudaFree(deviceTmpMinIdx_2);

}

/**/
