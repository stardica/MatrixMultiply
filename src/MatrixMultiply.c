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
#define PROGRAM_FILE "/home/stardica/Desktop/MatrixMultiply/src/add_numbers.cl"
#define KERNEL_FUNC "add_numbers"
#define ARRAY_SIZE 64

//macros
#define PRINT(...) printf("Print from the Macro: %p %p\n", __VA_ARGS__)
#define PRINANDTFLUSH 	printf("code ran here\n");\
						fflush(stdout);

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
cl_device_id create_device(void);
cl_program build_program(cl_context ctx, cl_device_id dev, const char* filename);


char *OpenCLProgram = "__kernel void add_numbers(__global float4* data, \
      __local float* local_result, __global float* group_result) {		\
																		\
   float sum;\
   float4 input1, input2, sum_vector;\
   uint global_addr, local_addr;\
																		\
   global_addr = get_global_id(0) * 2;\
   input1 = data[global_addr];\
   input2 = data[global_addr+1];\
   sum_vector = input1 + input2;\
\
   local_addr = get_local_id(0);\
   local_result[local_addr] = sum_vector.s0 + sum_vector.s1 +\
                              sum_vector.s2 + sum_vector.s3;\
   barrier(CLK_LOCAL_MEM_FENCE);\
\
   if(get_local_id(0) == 0) {\
      sum = 0.0f;\
      for(int i=0; i<get_local_size(0); i++) {\
         sum += local_result[i];\
      }\
      group_result[get_group_id(0)] = sum;\
   }\
}";



//Main///////////////////////////////
//Brute force matrix multiplication

int main(int argc, char *argv[]){

	if (MODE == 3){

		printf("---Stream Mode---\n\n");

		/* OpenCL structures */
		   cl_device_id device;
		   cl_context context;
		   cl_program program;
		   cl_kernel kernel;
		   cl_command_queue queue;
		   cl_int i, j, err;
		   size_t local_size, global_size;

		   /* Data and buffers */
		   float data[ARRAY_SIZE];
		   float sum[2], total, actual_sum;
		   cl_mem input_buffer, sum_buffer;
		   cl_int num_groups;

		   /* Initialize data */
		   for(i=0; i<ARRAY_SIZE; i++) {
		      data[i] = 1.0f*i;
		   }

		   /* Create device and context */
		   device = create_device();
		   context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
		   if(err < 0) {
		      perror("Couldn't create a context");
		      exit(1);
		   }

		   /* Build program */
		   program = build_program(context, device, PROGRAM_FILE);

		   /* Create data buffer */
		   global_size = 8;
		   local_size = 4;
		   num_groups = global_size/local_size;
		   input_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, ARRAY_SIZE * sizeof(float), data, &err);
		   sum_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, num_groups * sizeof(float), sum, &err);

		   if(err < 0) {
		      perror("Couldn't create a buffer");
		      exit(1);
		   };

		   /* Create a command queue */
		   queue = clCreateCommandQueue(context, device, 0, &err);
		   if(err < 0) {
		      perror("Couldn't create a command queue");
		      exit(1);
		   };

		   /* Create a kernel */
		   kernel = clCreateKernel(program, KERNEL_FUNC, &err);
		   if(err < 0) {
		      perror("Couldn't create a kernel");
		      exit(1);
		   };

		   /* Create kernel arguments */
		   err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_buffer);
		   err |= clSetKernelArg(kernel, 1, local_size * sizeof(float), NULL);
		   err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &sum_buffer);
		   if(err < 0) {
		      perror("Couldn't create a kernel argument");
		      exit(1);
		   }

		   /* Enqueue kernel */
		   err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
		   if(err < 0) {
		      perror("Couldn't enqueue the kernel");
		      exit(1);
		   }

		   /* Read the kernel's output */
		   err = clEnqueueReadBuffer(queue, sum_buffer, CL_TRUE, 0, sizeof(sum), sum, 0, NULL, NULL);
		   if(err < 0) {
		      perror("Couldn't read the buffer");
		      exit(1);
		   }

		   /* Check result */
		   total = 0.0f;
		   for(j=0; j<num_groups; j++) {
		      total += sum[j];
		   }
		   actual_sum = 1.0f * ARRAY_SIZE/2*(ARRAY_SIZE-1);
		   printf("Computed sum = %.1f.\n", total);
		   if(fabs(total - actual_sum) > 0.01*fabs(actual_sum))
		      printf("Check failed.\n");
		   else
		      printf("Check passed.\n");

		   /* Deallocate resources */
		   clReleaseKernel(kernel);
		   clReleaseMemObject(sum_buffer);
		   clReleaseMemObject(input_buffer);
		   clReleaseCommandQueue(queue);
		   clReleaseProgram(program);
		   clReleaseContext(context);

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

/* Find a GPU or CPU associated with the first available platform */
cl_device_id create_device() {

   cl_platform_id platform;
   cl_device_id dev;
   int err;

   /* Identify a platform */
   err = clGetPlatformIDs(1, &platform, NULL);
   if(err < 0) {
      perror("Couldn't identify a platform");
      exit(1);
   }

   /* Access a device */
   err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &dev, NULL);
   if(err == CL_DEVICE_NOT_FOUND) {
      err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &dev, NULL);
   }
   if(err < 0) {
      perror("Couldn't access any devices");
      exit(1);
   }

   return dev;
}

/* Create program from a file and compile it */
cl_program build_program(cl_context ctx, cl_device_id dev, const char* filename) {

   cl_program program;
   FILE *program_handle;
   char *program_buffer, *program_log;
   size_t program_size, log_size;
   int err;

   /* Read program file and place content into buffer */
   program_handle = fopen(filename, "r");
   if(program_handle == NULL) {
      perror("Couldn't find the program file");
      exit(1);
   }
   fseek(program_handle, 0, SEEK_END);
   program_size = ftell(program_handle);
   rewind(program_handle);
   program_buffer = (char*)malloc(program_size + 1);
   program_buffer[program_size] = '\0';
   fread(program_buffer, sizeof(char), program_size, program_handle);
   fclose(program_handle);

   /* Create program from file */
   program = clCreateProgramWithSource(ctx, 1, (const char**)&program_buffer, &program_size, &err);
   if(err < 0) {
      perror("Couldn't create the program");
      exit(1);
   }
   free(program_buffer);

   /* Build program */
   err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
   if(err < 0) {

      /* Find size of log and print to std output */
      clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
      program_log = (char*) malloc(log_size + 1);
      program_log[log_size] = '\0';
      clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, log_size + 1, program_log, NULL);
      printf("%s\n", program_log);
      free(program_log);
      exit(1);
   }

   return program;
}
