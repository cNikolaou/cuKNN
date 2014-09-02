#ifndef KNN_GPU_UTILS_
#define KNN_GPU_UTILS_

void say_hello();
void euclidean_distance(double *X, double *Y, int D, int Q, int N, 
                        int index, double *diff);
void compute_distance_gpu(double *data, double *queries, int D, int Q, int N,
                          double *dist);                        

#endif
