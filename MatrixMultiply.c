#define _GNU_SOURCE

#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <setjmp.h>
#include <sys/sysinfo.h>
#include <stddef.h>
#include <assert.h>
#include <sys/types.h>
#include <math.h>
#include <sched.h>
#include <unistd.h>

#include <CL/cl.h>

#include "rdtsc.h"
#include "kernelPaths.dat"


#define BEGIN_PARALLEL_SECTION 325
#define END_PARALLEL_SECTION 326

//SIZE sets height and width of matrix
//MODE 0 = Test code
//MODE 1 = Single thread matrix multiply
//MODE 2 = Multi thread matrix multiply (max number is 19 pthread limit)
//MODE 3 = Stream mode. Make sure to check memory settings.
//MODE 4 = OpenCL kernel precompile
//MODE 5 = OpenCL test code
//MODE 6 = Stream and multi-thread combined
// 0/1 HSAMODE changes the OpenCL runtime API calls to ours.


#define SIZE 16
#define MODE 3
#define NUM_THREADS 1
#define CHUNKSIZE 4
#define HSAMODE 0

//LOCALMEM = 1 puts the cl_mem buffer in the GPU's local memory.
//SYSMEM = 1 puts the cl_mem buffers in the system main memory hierarchy.
//CACHEDMEM = 1 caches the buffers?
//note: eighter or, don't active both the localmem and sysmem.

#define CACHEDMEM 0
#define LOCALMEM 1
#define SYSMEM 0

//configure global and work sizes for stream mode
#define GWS_0 16
#define GWS_1 16
#define LWS_0 8
#define LWS_1 8

//Kernel run path
char KERNEL[] = "/home/stardica/.local/share/Trash/files/Delete/CDA7919DoctoralResearch/MatrixMultiply/src/MatrixMultiply.cl.bin.GPU";

//1 if GPU 0 if CPU -1 if not set
int CPUGPUFLAG = 1;

//macros
#define PRINT(...) printf("Print from the Macro: %p %p\n", __VA_ARGS__)
#define PRINANDTFLUSH 	printf("code ran here\n");\
						fflush(stdout);

typedef int bool;
enum {false, true};

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

struct MatrixData{

	int id;
	int num_chunks;
	int Size;
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
void *ChunkMultiply(void *data);
void LoadMatrices(void);
void PrintMatrices(void);
cl_context CreateContext(void);
cl_command_queue CreateCommandQueue(cl_context context, cl_device_id *device);
void Cleanup(cl_context context, cl_command_queue commandQueue, cl_program program, cl_kernel kernel);
cl_program CreateProgram(cl_context context, cl_device_id device, const char* fileName);
bool SaveProgramBinary(cl_program program, cl_device_id device, char* fileName);
cl_program CreateProgramFromBinary(cl_context context, cl_device_id device, const char* fileName);

cpu_set_t  mask;

inline void assignToThisCore(int core_id)
{
    CPU_ZERO(&mask);
    CPU_SET(core_id, &mask);
    sched_setaffinity(0, sizeof(mask), &mask);
}

int main(int argc, char *argv[]){

	if (MODE == 5){

		printf("---OpenCL Test Code---\n\n");


		cl_int errNum;
		cl_uint numPlatforms;
		cl_platform_id *platforms = NULL;
		cl_uint numDevices;
		cl_device_id *devices = NULL;

		//platform info fields
		char vendor[1024], name[1024], version[1024];

		//device info fields
		size_t MAX_WORK_GROUP_SIZE;
		cl_ulong GLOBAL_MEM_CACHE_SIZE, GLOBAL_MEM_SIZE, LOCAL_MEM_SIZE, GLOBAL_MEM_CACHELINE_SIZE;
		cl_uint MAX_COMPUTE_UNITS, MAX_WORK_ITEM_DIMENSIONS;
		size_t MAX_WORK_ITEM_SIZES[3];
		char DEVICE_NAME[1024], DEVICE_VENDOR[1024], DEVICE_VERSION[1024], DRIVER_VERSION[1024], EXTENSIONS[2048];
		cl_device_mem_cache_type GLOBAL_MEM_CACHE_TYPE;


		//printf("Getting number of OpenCL Platforms...\n");
		errNum = clGetPlatformIDs(0, NULL, &numPlatforms);
		if (errNum != CL_SUCCESS)
		{
			printf("Failed to get number of OpenCL platforms.\n");
			return 0;
		}
		else
		{

			//printf("found %d.\n", numPlatforms);
		}

		//printf("Allocating space for the platform info...\n");
		platforms = (cl_platform_id *)malloc(numPlatforms*sizeof(cl_platform_id));

		printf("---Platform Info---\n");
		errNum = clGetPlatformIDs(numPlatforms, platforms, NULL);
		if (errNum != CL_SUCCESS)
		{
			printf("Failed to get platform info.\n");
			return 0;
		}
		else
		{
			clGetPlatformInfo (platforms[0], CL_PLATFORM_VENDOR, sizeof(vendor), vendor, NULL);
			clGetPlatformInfo (platforms[0], CL_PLATFORM_NAME, sizeof(name), name, NULL);
			clGetPlatformInfo (platforms[0], CL_PLATFORM_VERSION, sizeof(version), version, NULL);

			//printf("Got platform info.\n");
			printf("Vendor: \t%s\n", vendor);
			printf("Name:   \t%s\n", name);
			printf("Version:\t%s\n", version);
		}

		//printf("Getting number of devices...\n");
		errNum = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices);
		if (errNum != CL_SUCCESS)
		{
			printf("Failed to get number of devices.\n");
			return 0;
		}
		else
		{
	    	//printf("Found %d.\n", numDevices);
	    }

		//printf("Allocating space for device info...\n");
		devices = (cl_device_id*)malloc(numDevices * sizeof(cl_device_id));

		printf("\n---Device Info---");
		errNum = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, numDevices, devices, NULL);
		if (errNum != CL_SUCCESS)
		{
			printf("Failed to get device info.\n");
			return 0;
		}
		else
		{

			int i, j = 0;
			for (i = 0; i < numDevices; i++ )
			{
				printf("\nDevice ID: %d\n", i+1);
				clGetDeviceInfo(devices[i], CL_DEVICE_NAME, sizeof(DEVICE_NAME), DEVICE_NAME, NULL);
				clGetDeviceInfo(devices[i], CL_DEVICE_VENDOR, sizeof(DEVICE_VENDOR), DEVICE_VENDOR, NULL);
				clGetDeviceInfo(devices[i], CL_DEVICE_VERSION, sizeof(DEVICE_VERSION), DEVICE_VERSION, NULL);
				clGetDeviceInfo(devices[i], CL_DRIVER_VERSION, sizeof(DRIVER_VERSION), DRIVER_VERSION, NULL);
				clGetDeviceInfo(devices[i], CL_DEVICE_EXTENSIONS, sizeof(EXTENSIONS), EXTENSIONS, NULL);
				clGetDeviceInfo(devices[i], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(MAX_COMPUTE_UNITS), &MAX_COMPUTE_UNITS, NULL);
				clGetDeviceInfo(devices[i], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(GLOBAL_MEM_SIZE), &GLOBAL_MEM_SIZE, NULL);
				clGetDeviceInfo(devices[i], CL_DEVICE_LOCAL_MEM_SIZE, sizeof(LOCAL_MEM_SIZE), &LOCAL_MEM_SIZE, NULL);
				clGetDeviceInfo(devices[i], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(MAX_WORK_ITEM_DIMENSIONS), &MAX_WORK_ITEM_DIMENSIONS, NULL);
				clGetDeviceInfo(devices[i], CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(MAX_WORK_ITEM_SIZES), MAX_WORK_ITEM_SIZES, NULL);
				clGetDeviceInfo(devices[i], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(MAX_WORK_GROUP_SIZE), &MAX_WORK_GROUP_SIZE, NULL);
				clGetDeviceInfo(devices[i], CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, sizeof(GLOBAL_MEM_CACHE_SIZE), &GLOBAL_MEM_CACHE_SIZE, NULL);
				clGetDeviceInfo(devices[i], CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, sizeof(GLOBAL_MEM_CACHELINE_SIZE), &GLOBAL_MEM_CACHELINE_SIZE, NULL);
				clGetDeviceInfo(devices[i], CL_DEVICE_GLOBAL_MEM_CACHE_TYPE, sizeof(GLOBAL_MEM_CACHE_TYPE), &GLOBAL_MEM_CACHE_TYPE, NULL);


				printf("Device Name:\t%s\n", DEVICE_NAME);
				printf("Device Vendor:\t%s\n", DEVICE_VENDOR);
				printf("Device Version:\t%s\n", DEVICE_VERSION);
				printf("Driver Version:\t%s\n", DRIVER_VERSION);
				printf("EXTENSIONS:\t%s\n", EXTENSIONS);
				printf("Number of CUs:\t%d\n", MAX_COMPUTE_UNITS);
				printf("GMem:\t\t%lld (Bytes)\n", (long long) GLOBAL_MEM_SIZE);
				printf("GMem $ Size:\t%lld (Bytes)\n", (long long) GLOBAL_MEM_CACHE_SIZE);
				printf("GMem $ Line:\t%lld (Bytes)\n", (long long) GLOBAL_MEM_CACHELINE_SIZE);
				if(GLOBAL_MEM_CACHE_TYPE == CL_NONE)
				{
					printf("GMem $ Type:\tCL_NONE\n");
				}
				else if(GLOBAL_MEM_CACHE_TYPE == CL_READ_ONLY_CACHE)
				{
					printf("GMem $ Type:\tCL_READ_ONLY_CACHE\n");
				}

				else if(GLOBAL_MEM_CACHE_TYPE == CL_READ_WRITE_CACHE)
				{
					printf("GMem $ Type:\tCL_READ_WRITE_CACHE\n");
				}
				printf("LMem:\t\t%lld (Bytes)\n", (long long) LOCAL_MEM_SIZE);
				printf("Work Group Size:%d (Max)\n", (int) MAX_WORK_GROUP_SIZE);
				printf("Work Item Dim:\t%d (Max)\n", MAX_WORK_ITEM_DIMENSIONS);
				printf("Work Item Size:\t");
				for(j = 0; j < MAX_WORK_ITEM_DIMENSIONS; j ++)
				{
						if (j != (MAX_WORK_ITEM_DIMENSIONS -1))
						printf("%d, ", (int) MAX_WORK_ITEM_SIZES[j]);

						if (j == (MAX_WORK_ITEM_DIMENSIONS -1))
						printf("%d ", (int) MAX_WORK_ITEM_SIZES[j]);
				}
				printf("(Max)\n");

			}

				//printf("Got device info.\n");
		}


	}

	else if (MODE == 4){
		cl_context context = 0;
	    cl_command_queue commandQueue = 0;
	    cl_program program = 0;
	    cl_device_id device = 0;

	    //Create an OpenCL context on first available platform
	    context = CreateContext();
	    if (context == NULL)
	    {
	        printf("Failed to create OpenCL context.\n");
	        return 1;
	    }

	    //Create a command-queue on the first device available on the created context
	    commandQueue = CreateCommandQueue(context, &device);
	    if (commandQueue == NULL)
	    {
	    	printf("Failed to create commandQueue.\n");
	    	Cleanup(context, commandQueue, program, NULL);
	    	return 1;
	    }

	    // Create OpenCL program and store the binary for future use.
	    printf("Attempting to create kernel binary from source.\n");
	    program = CreateProgram(context, device, KERNELPATHIN);
	    if (program == NULL)
	    {
	    	printf("Failed to create Program");
	    	Cleanup(context, commandQueue, program, NULL);
	    	return 1;
	    }

	    printf("Kernel is saved.\n");
	    if (SaveProgramBinary(program, device, KERNELPATHOUT) == false)
	    {
	        printf("Failed to write program binary.\n");
	        Cleanup(context, commandQueue, program, NULL);
	        return 1;
	     }

	    //printf("---Done---");

	    //return 1;

	}
	else if (MODE == 3){

		//todo free remaining objects not passed to cleanup

		//profiling
		int write_bytes = 0;
		int read_bytes = 0;
		/*unsigned long long start_cycles, stop_cycles;
		unsigned long long start_setup, stop_setup;
		unsigned long long start_write, stop_write;
		unsigned long long start_read, stop_read;
		unsigned long long start_finalize, stop_finalize;
		struct timespec start_time_t, stop_time_t;*/


		printf("Stream Mode\n\n");
		//clock_gettime(CLOCK_MONOTONIC, &start_time_t);
		//start_cycles = rdtsc();


		int i;
		time_t t;
		srand((unsigned) time(&t));

	    // Create the two input vectors
	    printf("\nHostside malloc(s)\n");
	    fflush(stdout);
	    int *A = (int*)malloc(sizeof(int)*(SIZE*SIZE));
	    int *B = (int*)malloc(sizeof(int)*(SIZE*SIZE));
	    int *C = (int*)malloc(sizeof(int)*(SIZE*SIZE));

	    //profile
	    //bytes += 3 * sizeof(int)*(SIZE*SIZE);

	    printf("\nHostside mat init\n");
	    fflush(stdout);
	    for(i = 0; i < (SIZE*SIZE); i++) {
	        A[i] = B[i] = rand() % 10 + 1;;
	    }


	    //print matrix
    	printf("Matrix A[%d][%d]:\n", SIZE, SIZE);
	   	for(i = 0; i < (SIZE*SIZE); i++)
	    {
	   		printf("%3d ", A[i]);
	   		if(((i + 1) % SIZE) == 0)
	   			printf("\n");
	    }

	    //print matrix
	   	printf("\nMatrix B[%d][%d]:\n", SIZE, SIZE);
	    for(i = 0; i < (SIZE*SIZE); i++)
	    {
	    	printf("%3d ", B[i]);
	        if(((i + 1) % SIZE) == 0)
	        	printf("\n");
	    }


	    //syscall(STATS_RESET);


	    //Get platform and device information
	    cl_context context = 0;
	    cl_command_queue commandQueue = 0;
	    cl_program program = 0;
	    cl_device_id device = 0;
	    cl_kernel kernel = 0;
	    cl_uint err = 0;
	    //char *filepath = NULL;

	    //Create the context
	    printf("\nCreateContext\n");
	    fflush(stdout);
	    context = CreateContext();
	    if (context == NULL)
	    {
	    	printf("Failed to create OpenCL context.\n");
	    	return 1;
	    }

	   /* printf("\nEnd CreateContext\n");
	    fflush(stdout);*/

	    //Create a command-queue on the first device available on the created context
	    printf("\nCreateCommandQueue\n");
	    fflush(stdout);
	    commandQueue = CreateCommandQueue(context, &device);
	    if (commandQueue == NULL)
	    {
	    	printf("Failed to create command queue.\n");
	    	Cleanup(context, commandQueue, program, NULL);
	    	return 1;
	    }

	    //create the program from the binary
	    //program = CreateProgramFromBinary(context, device, "/home/stardica/Desktop/Kernels/vector.cl.bin.GPU");
	    //strcat(KERNELPATHOUT, ".GPU")
	    printf("\nCreateProgramFromBinary\n");
	    fflush(stdout);
	    program = CreateProgramFromBinary(context, device, KERNEL);
	    if (program == NULL)
	    {
	    	printf("Failed to load kernel binary,\n");
	    	Cleanup(context, commandQueue, program, NULL);
	    	return 1;
	    }



	    // Create OpenCL kernel
	    printf("\nclCreateKernel\n");
	    fflush(stdout);
	    kernel = clCreateKernel(program, "Matrix", NULL);
	    if (kernel == NULL)
	    {
	    	printf("Failed to create kernel.\n");
	    	Cleanup(context, commandQueue, program, NULL);
	    	return 1;
	    }

	    cl_mem a_mem_obj = 0;
	    cl_mem b_mem_obj = 0;
	    cl_mem c_mem_obj = 0;

  	    //Create memory buffers on the device for each vector

	    printf("\nclCreateBuffer(s)\n");
	    fflush(stdout);
	    if(LOCALMEM == 1 && CACHEDMEM == 0)
		{
			//this creates uncached buffers in the GPU's local memory
			#if M2S_CGM_OCL_SIM
			{
				a_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, (sizeof(int)*(SIZE*SIZE)), NULL, NULL, CL_FALSE);
				b_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, (sizeof(int)*(SIZE*SIZE)), NULL, NULL, CL_FALSE);
				c_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY, (sizeof(int)*(SIZE*SIZE)), NULL, NULL, CL_FALSE);

			}
			#else
			{
				a_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, (sizeof(int)*(SIZE*SIZE)), NULL, NULL);
				b_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, (sizeof(int)*(SIZE*SIZE)), NULL, NULL);
				c_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY, (sizeof(int)*(SIZE*SIZE)), NULL, NULL);
			}
			#endif
		}

		if(SYSMEM == 1 && CACHEDMEM == 0)
		{
			//this creates uncached buffers in the system memory
			#if M2S_CGM_OCL_SIM
			{
				a_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, (sizeof(int)*(SIZE*SIZE)), NULL, NULL, CL_FALSE);
				b_mem_obj = clCreateBuffer(context,CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, (sizeof(int)*(SIZE*SIZE)), NULL, NULL, CL_FALSE);
				c_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, (sizeof(int)*(SIZE*SIZE)), NULL, NULL, CL_FALSE);
			}
			#else
			{
				a_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, (sizeof(int)*(SIZE*SIZE)), NULL, NULL);
				b_mem_obj = clCreateBuffer(context,CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, (sizeof(int)*(SIZE*SIZE)), NULL, NULL);
				c_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, (sizeof(int)*(SIZE*SIZE)), NULL, NULL);
			}
			#endif
		}

		if(SYSMEM == 1 && CACHEDMEM == 1)
		{
			//this creates cached buffers in the system memory.
			#if M2S_CGM_OCL_SIM
			{
				a_mem_obj = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR, (sizeof(int)*(SIZE*SIZE)), NULL, NULL, CL_FALSE);
				b_mem_obj = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR, (sizeof(int)*(SIZE*SIZE)), NULL, NULL, CL_FALSE);
				c_mem_obj = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR, (sizeof(int)*(SIZE*SIZE)), NULL, NULL, CL_FALSE);
			}
			#else
			{
				a_mem_obj = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR, (sizeof(int)*(SIZE*SIZE)), NULL, NULL);
				b_mem_obj = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR, (sizeof(int)*(SIZE*SIZE)), NULL, NULL);
				c_mem_obj = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR, (sizeof(int)*(SIZE*SIZE)), NULL, NULL);
			}
			#endif
		}

	    if (a_mem_obj == NULL || b_mem_obj == NULL  || c_mem_obj == NULL)
	    {
	    	printf("Failed to create memory objects.\n");
	    	Cleanup(context, commandQueue, program, kernel);
	    	return 1;
	    }

	    //Copy the lists A and B to their respective memory buffers
	    printf("\nclEnqueueWriteBuffer(s)\n");
	    fflush(stdout);
	    write_bytes += 2 * sizeof(int)*(SIZE*SIZE);
	   // start_write = rdtsc();
	    clEnqueueWriteBuffer(commandQueue, a_mem_obj, CL_TRUE, 0, (sizeof(int)*(SIZE*SIZE)), A, 0, NULL, NULL);
	    clEnqueueWriteBuffer(commandQueue, b_mem_obj, CL_TRUE, 0, (sizeof(int)*(SIZE*SIZE)), B, 0, NULL, NULL);
	   // stop_write = rdtsc();


	    // Set the arguments of the kernel
	    int *size = (int *)SIZE;
	    printf("\nclSetKernelArg(s)\n");
	    fflush(stdout);
	    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&c_mem_obj);
	    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&a_mem_obj);
	    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&b_mem_obj);
	    err = clSetKernelArg(kernel, 3, sizeof(int), (void *)&size);
	    if (err != CL_SUCCESS)
	    {
	    	printf("Kernel args not set.\n");
	    	return 1;
	    }

	    // Execute the OpenCL kernel on the list
	    size_t GlobalWorkSize[2], LocalWorkSize[2];

	    //Rember that in OpenCL we need to express the globalWorkSize in
	    //terms of the total number of threads. The underlying OpenCL API
	    //will look at the globalWorkSize and divide by the localWorkSize
	    //to arrive at a 64 by 64 NDRange of 16 by 16 work groups.

	    GlobalWorkSize[0] = GWS_0;//SIZE*SIZE*SIZE; // Process the entire lists
	    GlobalWorkSize[1] = GWS_1;//SIZE*SIZE*SIZE; // Process the entire lists
	    LocalWorkSize[0] = LWS_0; //SIZE Divide work items into groups of 64
	    LocalWorkSize[1] = LWS_1; //SIZE Divide work items into groups of 64


	    //used null for local, lets OpenCL determine the best local size.
	    //err = clEnqueueNDRangeKernel(commandQueue, kernel, 2, NULL, GlobalWorkSize, LocalWorkSize, 0, NULL, NULL);
	    printf("\nclEnqueueNDRangeKernel\n");
	    fflush(stdout);
	    err = clEnqueueNDRangeKernel(commandQueue, kernel, 2, NULL, GlobalWorkSize, LocalWorkSize, 0, NULL, NULL);
	    if (err != CL_SUCCESS)
	    {
	    	printf("ND range not enqueued. Code: %d\n", err);
	    	return 1;
	    }


	    //Read the memory buffer C on the device to the local variable C
	    printf("\nclEnqueueReadBuffer\n");
	    fflush(stdout);
	    read_bytes += sizeof(int)*(SIZE*SIZE);
	    //start_read = rdtsc();
	    err = clEnqueueReadBuffer(commandQueue, c_mem_obj, CL_TRUE, 0, (sizeof(int)*(SIZE*SIZE)), C, 0, NULL, NULL);
	   // stop_read = rdtsc();
	    if (err != CL_SUCCESS)
	    {
	    	printf("Buffer not returned.\n");
	    	return 1;
	    }

	    //syscall(STATS_STOP);

	    //print matrix
	    printf("\nMatrix C[%d][%d] = A[%d][%d]*B[%d][%d]:\n", SIZE, SIZE, SIZE, SIZE, SIZE, SIZE);
	    for(i = 0; i < (SIZE*SIZE); i++)
	    {
	    	printf("%3d ", C[i]);
	        if(((i + 1) % SIZE) == 0)
	        printf("\n");
	    }

	    printf("\nHostside clean up\n");
	    fflush(stdout);
	    err = clFlush(commandQueue);
	    err = clFinish(commandQueue);
	    Cleanup(context, commandQueue, program, kernel);
	    err = clReleaseMemObject(a_mem_obj);
	    err = clReleaseMemObject(b_mem_obj);
	    err = clReleaseMemObject(c_mem_obj);
	    free(A);
	    free(B);
	    free(C);

	    //printf("---Done---");



	    /*stop_cycles = rdtsc();
	    clock_gettime(CLOCK_MONOTONIC, &stop_time_t);
	    printf("Total cycles = %llu\n", (stop_cycles - start_cycles));

	    long int time_s = stop_time_t.tv_nsec - start_time_t.tv_nsec;
	    printf("Approximate runtime (check) = %ld ms\n", (time_s/1000000));

	    printf("Bytes written %d\n", write_bytes);
	    printf("transfer cycles = %llu\n", (stop_write - start_write));
	    printf("start at = %llu\n", (start_write - start_cycles));

	    printf("Bytes read %d\n", read_bytes);
	    printf("transfer cycles = %llu\n", (stop_read - start_read));
	    printf("start at = %llu\n", (start_read - start_cycles));*/

	}
	else if (MODE == 2){

		printf("Multi Thread Mode\n");
		//cal this:
		//assignToThisCore(0);//assign to core 0,1,2,...

		unsigned long long a, b;
	    int i = 0;
	    int j = 0;
	    int k = 0;

		LoadMatrices();

		pthread_t tid[SIZE*SIZE];


		//printf("waiting\n");
		//start our threads
		a = rdtsc();
		syscall(BEGIN_PARALLEL_SECTION);

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
		for (i=0;i<NUM_THREADS;i++)
		{
			pthread_join(tid[i], NULL);
		}
		syscall(END_PARALLEL_SECTION);
		b = rdtsc();

		PrintMatrices();



		//printf("\nend clock Cycles: %llu\n", b);
		printf("\nDone. Number of clock Cycles: %llu\n", b-a);

	}
	else if (MODE == 1)
	{

		printf("Single Thread Mode\n\n");
		//unsigned long long a, b;
		//a = rdtsc();
		//time_t t;
		int i,j,k;

		//srand((unsigned) time(&t));

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

		//b = rdtsc();
		//printf("\nDone. Number of clock Cycles: %llu\n", b-a);
	}
	else if (MODE == 0)
	{
		printf("---Misc Tests---\n\n");

		printf("size of long long is %d\n", (int) sizeof(long long));
		printf("size of long is %d\n", (int) sizeof(long));
		printf("size of int is %d\n", (int) sizeof(int));
		printf("size of short is %d\n", (int) sizeof(short));
		printf("size of char * %d\n", (int) sizeof(char *));
		printf("size of unsigned int (word) %d\n", (int) sizeof(unsigned int));

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

		int mmu_page_size = 1 << 12;

		printf("mmu_papge_size = %d\n", mmu_page_size);


		//setjmp and longjmp fun
		/*jmp_buf environment;
		int i;

		i = setjmp(environment);
		printf("\n\nsetjmp returned = %d\n", i);

		printf("Env 1:\n");

		int x = 0;
		for(x = 0; x < 6; x++)
		{
			printf("  %x\n", environment[x]);
		}


		if (i < 3)
		{
			longjmp(environment, 3);
		}

		printf("longjmp finished with i = %d\n", i);*/


	}
	else
	{

		printf("---Invalid Mode Set---\n\n");

	}

	printf("\n---Done---\n");
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
	//rand() % 10 + 1;

	//fill mats a and b with random int between 1 and 10
	for (i=0;i<SIZE;i++){
		for(j=0;j<SIZE;j++){
			matA[i][j] = matB[i][j] = rand() % 10 + 1;
		}
	}

	//PrintMatrices();
}

void PrintMatrices(void){

	int i, j;

	//Display output//////////////////////////
	printf("Matrix A[%d][%d]:\n", SIZE, SIZE);
	for (i=0;i<SIZE;i++){
		for(j=0;j<SIZE;j++){
	 		printf("%3d ", matA[i][j]);
	 	}
	 	printf("\n");
	}

	printf("\n");

	 printf("Matrix B[%d][%d]:\n", SIZE, SIZE);
	 for (i=0;i<SIZE;i++){
	 	for(j=0;j<SIZE;j++){
	 		printf("%3d ", matB[i][j]);
	 	}
		printf("\n");
	 }

	 printf("\nMatrix C[%d][%d] = A[%d][%d]*B[%d][%d]:\n", SIZE, SIZE, SIZE, SIZE, SIZE, SIZE);
	 for (i=0;i<SIZE;i++){
	 	for(j=0;j<SIZE;j++){
	 		printf("%d ", matC[i][j]);
	 	}
	 printf("\n");
	 }
}

void *ChunkMultiply(void *data){

	struct MatrixData *RCData = data;
	int i = 0;
	int j = 0;
	int k = 0;
	//int SumOfProducts = 0;

	int col_num = RCData->ColumnNum;

	printf("my id is %d row is %d col is %d\n", RCData->id, RCData->RowNum, RCData->ColumnNum);

	printf("size %d\n", RCData->Size);

	//multiply mats/////////////////////////
	for (i=0;i<RCData->Size;i++)
	{
		for(j=0;j<RCData->Size;j++)
		{
			for(k=0;k<RCData->Size;k++)
			{
				//printf("matC[%d][%d] = matC[%d][%d] + (matA[%d][%d] * matB[%d][%d])\n",
				//		RCData->RowNum, RCData->ColumnNum, RCData->RowNum, RCData->ColumnNum, RCData->RowNum, k, k, RCData->ColumnNum);
				matC[RCData->RowNum][RCData->ColumnNum] = matC[RCData->RowNum][RCData->ColumnNum] + (matA[RCData->RowNum][k] * matB[k][RCData->ColumnNum]);
				//matC[i][j] = matC[i][j] + (matA[i][k] * matB[k][j]);
			}

			RCData->ColumnNum++;
		}

		RCData->ColumnNum = col_num;
		RCData->RowNum++;
	}

	pthread_exit(0);
}


void *RowColumnMultiply(void *data){


	struct RowColumnData *RCData = data;
	int i, SumOfProducts = 0;

	//printf("from thread Row %d Col %d\n", RCData->RowNum, RCData->ColumnNum);

	for(i=0;i<SIZE;i++)
	{
		SumOfProducts += matA[RCData->RowNum][i] * matB[i][RCData->ColumnNum];
	}

	//assign the sum to its coordinate
	matC[RCData->RowNum][RCData->ColumnNum] = SumOfProducts;
	free(data);
	pthread_exit(0);
}

cl_context CreateContext() {
    cl_int errNum;
    cl_uint numPlatforms;
    cl_platform_id firstPlatformId;
    cl_context context = NULL;

    // First, select an OpenCL platform to run on.  For this example, we
    // simply choose the first available platform.  Normally, you would
    // query for all available platforms and select the most appropriate one.



    errNum = clGetPlatformIDs(1, &firstPlatformId, &numPlatforms);



    if (errNum != CL_SUCCESS || numPlatforms <= 0)
    {
        printf("Failed to find any OpenCL platforms.\n");
        return NULL;
    }



    // Next, create an OpenCL context on the platform.  Attempt to
    // create a GPU-based context, and if that fails, try to create
    // a CPU-based context.
    cl_context_properties contextProperties[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)firstPlatformId, 0 };


    if (CPUGPUFLAG == 1)
    {
    	context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_GPU, NULL, NULL, &errNum);
    	if (errNum != CL_SUCCESS)
    	{
    		printf("Could not create GPU context.\n");
    		return NULL;
    	}
    }


    if (CPUGPUFLAG == 0)
    {
    	context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_CPU, NULL, NULL, &errNum);
    	if (errNum != CL_SUCCESS)
    	{
    		printf("Failed to create an OpenCL GPU or CPU context.\n");
    		return NULL;
    	}

    }


    return context;
}

cl_command_queue CreateCommandQueue(cl_context context, cl_device_id *device)
{
    cl_int errNum;
    cl_device_id *devices;
    cl_command_queue commandQueue = NULL;
    size_t deviceBufferSize = -1;

    // First get the size of the devices buffer
    errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &deviceBufferSize);
    if (errNum != CL_SUCCESS)
    {
        printf("Failed call to clGetContextInfo(...,GL_CONTEXT_DEVICES,...)\n");
        return NULL;
    }

    if (deviceBufferSize <= 0)
    {
        printf("No devices available.\n");
        return NULL;
    }

    // Allocate memory for the devices buffer
    devices = (cl_device_id *) malloc(deviceBufferSize / sizeof(cl_device_id));
    errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, deviceBufferSize, devices, NULL);
    if (errNum != CL_SUCCESS)
    {
        free(devices);
        printf("Failed to get device IDs");
        return NULL;
    }

    // In this example, we just choose the first available device.  In a
    // real program, you would likely use all available devices or choose
    // the highest performance device based on OpenCL device queries
    commandQueue = clCreateCommandQueue(context, devices[0], 0, NULL);
    if (commandQueue == NULL)
    {
        free(devices);
        printf("Failed to create commandQueue for device 0");
        return NULL;
    }

    *device = devices[0];
    free(devices);
    return commandQueue;
}

void Cleanup(cl_context context, cl_command_queue commandQueue, cl_program program, cl_kernel kernel) {


    if (commandQueue != 0)
        clReleaseCommandQueue(commandQueue);

    if (kernel != 0)
        clReleaseKernel(kernel);

    if (program != 0)
        clReleaseProgram(program);

    if (context != 0)
        clReleaseContext(context);

}

cl_program CreateProgram(cl_context context, cl_device_id device, const char* fileName) {

	cl_int errNum;
    cl_program program;

    char *buffer;
    long length = 0;
    //char temp;
    FILE *fp;

    fp = fopen(fileName, "r");
    if (fp == NULL)
    {
        printf("Failed to open input file.\n");
        return NULL;
    }

    fseek(fp, 0L, SEEK_END);
    //apparently using ftell to get file size of a text file is bad.
    length = ftell(fp);
    if (length < 0){
    	printf("Error getting file size.\n");
    	return NULL;
    }
    fseek(fp, 0L, SEEK_SET);

    buffer = (char *) malloc(length + 1);

    int elems = 0;
    elems = fread(buffer, 1, length, fp);
    assert(elems);
    //ftell is bad. Sometimes you get garbage at the end of the string.
    //add 0 to the end to terminate the string correctly.
    buffer[length] = 0;
    printf("%s\n", buffer);
    fclose(fp);

    program = clCreateProgramWithSource(context, 1, (const char**)&buffer, NULL, NULL);
    if (program == NULL)
    {
        printf("Failed to create CL program from source.\n");
        return NULL;
    }

    errNum = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (errNum != CL_SUCCESS)
    {
        // Determine the reason for the error
        char buildLog[16384];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buildLog), buildLog, NULL);

        printf("Error in kernel\n\n");
        printf("%s\n", buildLog);
        clReleaseProgram(program);
        return NULL;
    }

    return program;
}

bool SaveProgramBinary(cl_program program, cl_device_id device, char* fileName){

	cl_uint numDevices = 0;
    cl_int errNum;
    cl_device_id *devices;
    size_t *programBinarySizes;
    unsigned char **programBinaries;
    cl_uint i;

    // 1 - Query for number of devices attached to program
    errNum = clGetProgramInfo(program, CL_PROGRAM_NUM_DEVICES, sizeof(cl_uint), &numDevices, NULL);
    if (errNum != CL_SUCCESS)
    {
        printf("Error querying for number of devices.\n");
        return false;
    }

    // 2 - Get all of the Device IDs
    devices = (cl_device_id *) malloc(sizeof(cl_device_id[numDevices]));
    errNum = clGetProgramInfo(program, CL_PROGRAM_DEVICES, sizeof(cl_device_id) * numDevices, devices, NULL);

    if (errNum != CL_SUCCESS)
    {
    	printf("Error querying for devices.\n");
        free(devices);
        return false;
    }

    // 3 - Determine the size of each program binary
    programBinarySizes = (size_t *) malloc(sizeof(size_t[numDevices]));
    errNum = clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, sizeof(size_t) * numDevices, programBinarySizes, NULL);
    if (errNum != CL_SUCCESS)
    {
    	printf("Error querying for program binary sizes.\n");
    	free(devices);
    	free(programBinarySizes);
        return false;
    }

    //unsigned char **programBinaries = new unsigned char*[numDevices];
    programBinaries = (unsigned char **) malloc(sizeof(unsigned char *[numDevices]));

    for (i = 0; i < numDevices; i++)
    {

    	//programBinaries[i] = new unsigned char[programBinarySizes[i]];
    	programBinaries[i] = malloc(sizeof(unsigned char[programBinarySizes[i]]));

    }

    // 4 - Get all of the program binaries
    errNum = clGetProgramInfo(program, CL_PROGRAM_BINARIES, sizeof(unsigned char*) * numDevices, programBinaries, NULL);
    if (errNum != CL_SUCCESS)
    {
    	printf("Error querying for program binaries\n");
    	free(devices);
    	free(programBinarySizes);
        cl_uint i;
    	for (i = 0; i < numDevices; i++)
        {
            free(programBinaries[i]);
        }
        free(programBinaries);
        return false;
    }

    // 5 - Finally store the binaries for the device requested out to disk for future reading.

    if(CPUGPUFLAG == 1){
    	strcat(fileName, ".GPU");
    }
    else if (CPUGPUFLAG == 0){
    	strcat(fileName, ".CPU");
    }

    for (i = 0; i < numDevices; i++)
    {
        // Store the binary just for the device requested.  In a scenario where
        // multiple devices were being used you would save all of the binaries out here.
        if (devices[i] == device)
        {
            FILE *fp = fopen(fileName, "wb");
            fwrite(programBinaries[i], 1, programBinarySizes[i], fp);
            fclose(fp);
            break;
        }
    }

    // Cleanup
    free(devices);
    free(programBinarySizes);

    for (i = 0; i < numDevices; i++)
    {
    	free(programBinaries[i]);
    }
    free(programBinaries);
    return true;
}

cl_program CreateProgramFromBinary(cl_context context, cl_device_id device, const char* fileName)
{
    FILE *fp = fopen(fileName, "rb");
    if (fp == NULL)
    {
        return NULL;
    }

    // Determine the size of the binary
    size_t binarySize;
    fseek(fp, 0, SEEK_END);
    binarySize = ftell(fp);
    rewind(fp);

    unsigned char *programBinary;
    programBinary = (unsigned char *) malloc(sizeof(unsigned char[binarySize]));

    int elems = 0;
    elems = fread(programBinary, 1, binarySize, fp);
    assert(elems);
    fclose(fp);

    cl_int errNum = 0;
    cl_int binaryStatus;
    cl_program program;

    program = clCreateProgramWithBinary(context, 1, &device, &binarySize, (const unsigned char**)&programBinary, &binaryStatus,&errNum);

    if (errNum != CL_SUCCESS)
    {
        printf("Error loading program binary.\n");
        return NULL;
    }

    if (binaryStatus != CL_SUCCESS)
    {
        printf("Invalid binary for device,\n");
        return NULL;
    }


    errNum = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (errNum != CL_SUCCESS)
    {
        // Determine the reason for the error
        char buildLog[16384];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buildLog), buildLog, NULL);

        printf("CreateProgramFromBinary(): Error in kernel.\n");
        //printf("%s\n", buildLog);
        clReleaseProgram(program);
        return NULL;
    }

    free(programBinary);
    return program;
}


