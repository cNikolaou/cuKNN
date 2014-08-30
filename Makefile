CC=gcc -O4
NVCC=nvcc -O4
LIB_PATH=-L/usr/local/cuda-6.5/lib64

all: knnTest.o knns.o knn_gpu_utils.o

	$(CC) $(LIB_PATH) knnTest.o knns.o knn_gpu_utils.o -lcuda -lcudart -o knnTest 

knnTest.o: knnTest.c utils.h knns.h

	$(CC) knnTest.c -c

knns.o: knns.c utils.h knns.h knn_gpu_utils.h

	$(CC) knns.c -c 

knn_gpu_utils.o: knn_gpu_utils.cu knn_gpu_utils.h
	
	$(NVCC) knn_gpu_utils.cu -c 

clean:
	rm -f *.o *.out *~
	rm -f *.bin
