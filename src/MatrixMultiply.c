#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include "rdtsc.h"
#include <CL/cl.h>
#include <math.h>


//SIZE sets height and width of matrix
//MODE 0 = Test code
//MODE 1 = single thread Matrix Multiply
//MODE 2 = Multi thread Matrix Multiply
//MODE 3 = Stream Mode Matrix Multiply
//MODE 4 = OpenCL Kernel Precompile

#define SIZE 3
#define MODE 4

#define KERNELPATHIN	"/home/stardica/Desktop/Kernels/hello.cl"
#define KERNELPATHOUT	"/home/stardica/Desktop/Kernels/hello.cl.bin"

//macros
#define PRINT(...) printf("Print from the Macro: %p %p\n", __VA_ARGS__)
#define PRINANDTFLUSH 	printf("code ran here\n");\
						fflush(stdout);

typedef int bool;
enum { false, true };

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
cl_context CreateContext(void);
cl_command_queue CreateCommandQueue(cl_context context, cl_device_id *device);
void Cleanup(cl_context context, cl_command_queue commandQueue, cl_program program);
cl_program CreateProgram(cl_context context, cl_device_id device, const char* fileName);
bool SaveProgramBinary(cl_program program, cl_device_id device, const char* fileName);


//Main///////////////////////////////
//Brute force matrix multiplication

int main(int argc, char *argv[]){

	if (MODE == 4){

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
	    	Cleanup(context, commandQueue, program);
	    	return 1;
	    }

	    // Create OpenCL program and store the binary for future use.
	    printf("Attempting to create kernel binary from source.\n");
	    program = CreateProgram(context, device, KERNELPATHIN);
	    if (program == NULL)
	    {
	    	printf("Failed to create Program");
	    	Cleanup(context, commandQueue, program);
	    	return 1;
	    }

	    printf("Kernel is saved.\n");
	    if (SaveProgramBinary(program, device, KERNELPATHOUT) == false)
	    {
	        printf("Failed to write program binary.\n");
	        Cleanup(context, commandQueue, program);
	        return 1;
	     }

	    printf("---Done---");

	    return 1;


	}


	else if (MODE == 3){

		printf("---Stream Mode---\n\n");

		return 1;

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

    context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_GPU, NULL, NULL, &errNum);

    if (errNum != CL_SUCCESS)
    {
        printf("Could not create GPU context, trying CPU.\n");

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

void Cleanup(cl_context context, cl_command_queue commandQueue, cl_program program){


	if (commandQueue != 0)
		clReleaseCommandQueue(commandQueue);

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
    fread(buffer, 1, length, fp);
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

        printf("Error in kernel\n");
        //printf("%s\n", buildLog);
        clReleaseProgram(program);
        return NULL;
    }

    return program;
}

bool SaveProgramBinary(cl_program program, cl_device_id device, const char* fileName){

	cl_uint numDevices = 0;
    cl_int errNum;
    cl_device_id *devices;
    size_t *programBinarySizes;
    unsigned char **programBinaries;

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
    cl_uint i;
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
    //cl_uint i;
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
