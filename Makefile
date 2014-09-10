CC=gcc -O4
NVCC=nvcc -O4
LIB_PATH=-L/usr/local/cuda-6.5/lib64

INC_PATH=src
INC_LIST=utils.h knns.h knn_gpu_utils.h

SRC_PATH=src

OBJ_PATH=obj
OBJ_LIST=knnTest.o knns.o knn_gpu_utils.o

all: directories $(OBJ_PATH)/knnTest.o $(OBJ_PATH)/knns.o $(OBJ_PATH)/knn_gpu_utils.o

	$(CC) $(LIB_PATH) $(OBJ_PATH)/knnTest.o $(OBJ_PATH)/knns.o $(OBJ_PATH)/knn_gpu_utils.o -lcuda -lcudart -o knnTest 

directories:
	mkdir -p $(OBJ_PATH)

$(OBJ_PATH)/knnTest.o: src/knnTest.c include/utils.h include/knns.h

	$(CC) src/knnTest.c -c -o $(OBJ_PATH)/knnTest.o

$(OBJ_PATH)/knns.o: src/knns.c include/knns.h include/knn_gpu_utils.h

	$(CC) src/knns.c -c -o $(OBJ_PATH)/knns.o

$(OBJ_PATH)/knn_gpu_utils.o: src/knn_gpu_utils.cu include/knn_gpu_utils.h
	
	$(NVCC) src/knn_gpu_utils.cu -c -o $(OBJ_PATH)/knn_gpu_utils.o

clean:
	rm -f *.o *.out *~
	rm -f *.bin
