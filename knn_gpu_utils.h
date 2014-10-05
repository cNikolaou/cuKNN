/**
 *
 * Definition of functions to compute the distance and the k nearest
 * neighbors while using a GPU.
 *
 * More information are available in knn_gpu_utils.cu file, where the 
 * implementations are provided.
 *
 *
 * Author: Christos Nikolaou
 * Date: August 2014
 *
 */

#ifndef KNN_GPU_UTILS_
#define KNN_GPU_UTILS_
/*
void euclidean_distance(double *X, double *Y, int D, int Q, int N, 
                        int index, double *diff);
*/
void compute_distance_gpu(const double *data, const double *queries, 
                          const int D, const int Q, const int N,
                          double *dist);         
void selection_gpu(double* dist, double* NNdist, 
                   int* NNidx, int N, int Q, int k);

#endif
