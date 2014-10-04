#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

#include "utils.h"
#include "knns.h"

void random_initialization(knn_struct *set, int cal){

  int i = 0;
  int n = set->leading_dim;
  int m = set->secondary_dim;
  double *tmp_set = set->data;

  srand(0);
  //srand(cal*time(NULL));
  /*Generate random floating points [-50 50]*/
  for(i=0; i<m*n; i++){
    tmp_set[i] = 100 * (double)rand() / RAND_MAX - 50; 
  }

}

void save_d(double* data, char* file, int N, int M){

  int i = 0,j = 0;
  FILE *outfile;
  
  printf("Saving data to file: %s\n", file);

  if((outfile=fopen(file, "wb")) == NULL){
    printf("Can't open output file");
  }

  fwrite(data, sizeof(double), N*M, outfile);

  fclose(outfile);

}

void save_int(int* data, char* file, int N, int M){

  int i = 0,j = 0;
  FILE *outfile;
  
  printf("Saving data to file: %s\n", file);

  if((outfile=fopen(file, "wb")) == NULL){
    printf("Can't open output file");
  }

  fwrite(data, sizeof(int), N*M, outfile);

  fclose(outfile);

}

void clean(knn_struct* d){

  free(d->data);
}

int main(int argc, char **argv){

  struct timeval first, second, lapsed;
  struct timezone tzp;

  int numObjects = atoi(argv[1]);
  int numDim = atoi(argv[2]);
  int numQueries = atoi(argv[3]);
  int k = atoi(argv[4]);

  char *dataset_file = "training_set_gpu.bin";
  char *query_file = "query_set_gpu.bin";
  char *KNNdist_file = "KNNdist_gpu.bin";
  char *KNNidx_file = "KNNidx_gpu.bin" ;

  printf("objects: %d\n", numObjects);
  printf("dimentions: %d\n", numDim);
  printf("queries: %d\n", numQueries);
  printf("k: %d\n", k);

  knn_struct training_set;
  knn_struct query_set;
  double *NNdist;
  int *NNidx;

  training_set.leading_dim = numDim;
  training_set.secondary_dim = numObjects;
  query_set.leading_dim = numDim;
  query_set.secondary_dim = numQueries;

  /*======== Memory allocation ======*/
  training_set.data = (double*)malloc(numObjects*numDim*sizeof(double));
  query_set.data = (double*)malloc(numQueries*numDim*sizeof(double));
  NNdist = (double*)malloc(numQueries*k*sizeof(double));
  NNidx = (int*)malloc(numQueries*k*sizeof(int));


  /*======== initialize =========*/
  random_initialization(&training_set, 1);
  random_initialization(&query_set, 2);

  gettimeofday(&first, &tzp);

  knns(&query_set, &training_set, NNdist, NNidx, k);

  gettimeofday(&second, &tzp);

  if(first.tv_usec>second.tv_usec){
    second.tv_usec += 1000000;
    second.tv_sec--;
  }
  
  lapsed.tv_usec = second.tv_usec - first.tv_usec;
  lapsed.tv_sec = second.tv_sec - first.tv_sec;

  printf("Time elapsed: %d, %d s\n", lapsed.tv_sec, lapsed.tv_usec); 


  save_d(query_set.data, query_file, numQueries, numDim);
  save_d(training_set.data, dataset_file, numObjects, numDim);
  save_d(NNdist, KNNdist_file, k, numQueries);
  save_int(NNidx, KNNidx_file, k, numQueries);

  /*===== clean memory ========*/
  clean(&training_set);
  clean(&query_set);
  free(NNdist);
  free(NNidx);

}




