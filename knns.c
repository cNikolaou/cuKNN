/**
 * 
 * Modified by: Christos Nikolaou
 * Date: August 2014
 *
*/

#include <stdio.h>
#include <stdlib.h>

#include "omp.h"

#include "utils.h"
#include "knn_gpu_utils.h"

/*
 * This function has been changed to use a parallel version
 *
 */

/*
double euclidean_distance(double *X, double *Y, int N){

  int i = 0;
  double dst = 0;

  for(i=0; i<N; i++){
    double tmp = (X[i] - Y[i]);
    dst += tmp * tmp;
  }

  return(dst);
}
*/

void compute_distance(knn_struct* queries, knn_struct* dataset, double* dist) {

  int Q = queries->secondary_dim;
  int N = dataset->secondary_dim;
  int D = dataset->leading_dim;

  double *data = dataset->data;
  double *query = queries->data;

  compute_distance_gpu(data, query, D, Q, N, dist);

}

int findMax(double* X, int k){

  int i=0;
  int maxidx = 0;
  double maxval = X[0];

  for(i=1; i<k; i++){

    if(maxval<X[i]){
      maxval = X[i];
      maxidx = i;
    }
  }

  return(maxidx);

}

void kselect(double* dist, double* NNdist, int* NNidx, int N, int k) {


  int i = 0;

    for(i=0; i<k; i++){
      NNdist[i] = dist[i];
      NNidx[i] = i;
    }

    int maxidx = findMax(NNdist, k);

  for(i=k; i<N; i++){

    if(NNdist[maxidx]>dist[i]){
      NNdist[maxidx] = dist[i];
      NNidx[maxidx] = i;
      maxidx = findMax(NNdist, k);
    }
  }

}

void selection(double* dist, double* NNdist, int* NNidx, int N, int Q, int k) {

selection_gpu(dist, NNdist, NNidx, N, Q, k);

/*
  int i = 0, j = 0;

  for(i=0; i<Q; i++){
    kselect(&dist[i*N], &NNdist[i*k], &NNidx[i*k], N, k);
  }
*/
}

void knns(knn_struct* queries, knn_struct* dataset, double *NNdist, 
          int *NNidx, int k) {

  double *dist;
  int q = queries->secondary_dim;
  int n = dataset->secondary_dim;

  dist = (double*)malloc(n*q*sizeof(double));

  compute_distance(queries, dataset, dist);

  selection(dist, NNdist, NNidx, n, q, k);

  free(dist);
}





