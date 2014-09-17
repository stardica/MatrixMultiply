#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include "rdtsc.h"
#include <CL/cl.h>


//SIZE sets height and width of matrix
//MODE 0 = Test code
//MODE 1 = single thread Matrix Multiply
//MODE 2 = Multi thread Matrix Multiply
#define SIZE 2
#define MODE 3

//macros
 #define PRINT(...) printf("Print from the Macro: %p %p\n", __VA_ARGS__)

//objects
struct Object{
	char *string;
	void (*print_me)(void *ptr);
	struct Object *next;

};

struct Object object;
struct Object *object_ptr;

struct RowColumnData{
	int RowNum;
	int ColumnNum;
};

//Matrices
int matA[SIZE][SIZE];
int matB[SIZE][SIZE];
int matC[SIZE][SIZE];

//function declarations
void print_me(char *string);
void *RowColumnMultiply(void *data);
void LoadMatrices(void);
void PrintMatrices(void);


//Main///////////////////////////////
//Brute force matrix multiplication

int main(int argc, char *argv[]){

	if (MODE == 3){
	    int i, j;
	    char* value;
	    size_t valueSize;
	    cl_uint platformCount;
	    cl_platform_id* platforms;
	    cl_uint deviceCount;
	    cl_device_id* devices;
	    cl_uint maxComputeUnits;

	    // get all platforms
	    clGetPlatformIDs(0, NULL, &platformCount);
	    platforms = (cl_platform_id*) malloc(sizeof(cl_platform_id) * platformCount);
	    clGetPlatformIDs(platformCount, platforms, NULL);

	    for (i = 0; i < platformCount; i++) {

	        // get all devices
	        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &deviceCount);
	        devices = (cl_device_id*) malloc(sizeof(cl_device_id) * deviceCount);
	        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, deviceCount, devices, NULL);

	        // for each device print critical attributes
	        for (j = 0; j < deviceCount; j++) {

	            // print device name
	            clGetDeviceInfo(devices[j], CL_DEVICE_NAME, 0, NULL, &valueSize);
	            value = (char*) malloc(valueSize);
	            clGetDeviceInfo(devices[j], CL_DEVICE_NAME, valueSize, value, NULL);
	            printf("%d. Device: %s\n", j+1, value);
	            free(value);

	            // print hardware device version
	            clGetDeviceInfo(devices[j], CL_DEVICE_VERSION, 0, NULL, &valueSize);
	            value = (char*) malloc(valueSize);
	            clGetDeviceInfo(devices[j], CL_DEVICE_VERSION, valueSize, value, NULL);
	            printf(" %d.%d Hardware version: %s\n", j+1, 1, value);
	            free(value);

	            // print software driver version
	            clGetDeviceInfo(devices[j], CL_DRIVER_VERSION, 0, NULL, &valueSize);
	            value = (char*) malloc(valueSize);
	            clGetDeviceInfo(devices[j], CL_DRIVER_VERSION, valueSize, value, NULL);
	            printf(" %d.%d Software version: %s\n", j+1, 2, value);
	            free(value);

	            // print c version supported by compiler for device
	            clGetDeviceInfo(devices[j], CL_DEVICE_OPENCL_C_VERSION, 0, NULL, &valueSize);
	            value = (char*) malloc(valueSize);
	            clGetDeviceInfo(devices[j], CL_DEVICE_OPENCL_C_VERSION, valueSize, value, NULL);
	            printf(" %d.%d OpenCL C version: %s\n", j+1, 3, value);
	            free(value);

	            // print parallel compute units
	            clGetDeviceInfo(devices[j], CL_DEVICE_MAX_COMPUTE_UNITS,
	            sizeof(maxComputeUnits), &maxComputeUnits, NULL);
	            printf(" %d.%d Parallel compute units: %d\n", j+1, 4, maxComputeUnits);

	        }

	        free(devices);

	    }

	    free(platforms);

	}
	else if (MODE == 2){

		printf("---Multi Thread Mode---\n\n");
		int i, j, k;

		LoadMatrices();

		pthread_t tid[SIZE*SIZE];

		//start our threads
		k=0;
		for(i=0;i<SIZE;i++){
			for(j=0;j<SIZE;j++){
				struct RowColumnData *RCData = (struct RowColumnData *) malloc(sizeof(struct RowColumnData));
				RCData->RowNum = i;
				RCData->ColumnNum = j;
				//printf("Thread create %d Row %d Col %d\n", k, RCData->RowNum, RCData->ColumnNum);
				pthread_create(&tid[k], NULL, RowColumnMultiply, RCData);
				k++;
			}
		}

		//Join threads////////////////////////////
		for (i=0;i<(SIZE*SIZE);i++){
			pthread_join(tid[i], NULL);
			//printf("Thread join %d", i);
		}

		PrintMatrices();
	}
	else if (MODE == 1){

		printf("---Single Thread Mode---\n\n");
		unsigned long long a, b;
		a = rdtsc();
		time_t t;
		int i,j,k;

		srand((unsigned) time(&t));

		LoadMatrices();

		//multiply mats/////////////////////////
		for (i=0;i<SIZE;i++){
			for(j=0;j<SIZE;j++){
				for(k=0;k<SIZE;k++){
					matC[i][j] = matC[i][j] + (matA[i][k] * matB[k][j]);
					}
			}
		}

		PrintMatrices();

		b = rdtsc();
		printf("\nDone. Number of clock Cycles: %llu\n", b-a);
	}
	else if (MODE == 0)
	{
		printf("---Misc Tests---\n\n");

		printf("size of long long is %d\n", sizeof(long long));
		printf("size of long is %d\n", sizeof(long));
		printf("size of int is %d\n", sizeof(int));
		printf("size of short is %d\n", sizeof(short));

		char *string = "test string";
		printf("Here is the string 1: \"%s\"\n", string);

		//Using the struct
		//set string variable and point to print_me.
		object.string = strdup(string);
		object.print_me = (void (*)(void *)) print_me;

		//use of print_me
		object.print_me(object.string);

		//pointer fun
		struct Object *ptr = &object;
		printf("this is the value of the pointer to struct object: %p\n", ptr);
		object.next=&object;
		printf("this is the value of the pointer to struct object: %p\n", object.next);
		object_ptr = &object;
		object_ptr->next = &object;
		printf("this is the value of the pointer to struct object: %p\n", object_ptr->next);

		//Macro fun
		PRINT(ptr, ptr);
		PRINT(object.next, object.next);
		PRINT(object_ptr->next, object_ptr->next);

		//make sure the code ran all the way through.
		printf("done\n");

	}
	else
	{

		printf("---Invalid Mode Set---\n\n");

	}
	return 1;
}

void print_me(char *string)
{

	printf("Here is the string: \"%s\"\n", string);

}

void LoadMatrices(void){

	int i, j;

	time_t t;
	srand((unsigned) time(&t));

	//fill mats a and b with random int between 1 and 10
	for (i=0;i<SIZE;i++){
		for(j=0;j<SIZE;j++){
			matA[i][j] = rand() % 10 + 1;
		}
	}

	for (i=0;i<SIZE;i++){
		for(j=0;j<SIZE;j++){
			matB[i][j] = rand() % 10 + 1;
		}
	}
}

void PrintMatrices(void){

	int i, j;

	//Display output//////////////////////////
	printf("Matrix A[][]:\n");
	for (i=0;i<SIZE;i++){
		for(j=0;j<SIZE;j++){
	 		printf("%d ", matA[i][j]);
	 	}
	 	printf("\n");
	}

	 printf("\nMatrix B[][]:\n");
	 for (i=0;i<SIZE;i++){
	 	for(j=0;j<SIZE;j++){
	 		printf("%d ", matB[i][j]);
	 	}
		printf("\n");
	 }

	 printf("\nMatrix C[][] = A[][]*B[][]:\n");
	 for (i=0;i<SIZE;i++){
	 	for(j=0;j<SIZE;j++){
	 		printf("%d ", matC[i][j]);
	 	}
	 printf("\n");
	 }
}

void *RowColumnMultiply(void *data){


	struct RowColumnData *RCData = data;
	int i, SumOfProducts = 0;

	//printf("from thread Row %d Col %d\n", RCData->RowNum, RCData->ColumnNum);
	for(i=0;i<SIZE;i++){
		SumOfProducts += matA[RCData->RowNum][i] * matB[i][RCData->ColumnNum];
	   }

	//assign the sum to its coordinate
	matC[RCData->RowNum][RCData->ColumnNum] = SumOfProducts;
	pthread_exit(0);
}
