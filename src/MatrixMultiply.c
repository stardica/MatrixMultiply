#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include "rdtsc.h"

#define SIZE 4
#define MODE 2

//function declarations
void print_me(char *string);

//objects
struct Object{

	char *string;
	void (*print_me)(void *ptr);
	struct Object *next;

};

struct Object object;
struct Object *object_ptr;

int matA[SIZE][SIZE];
int matB[SIZE][SIZE];
int matC[SIZE][SIZE];


//Main///////////////////////////////
//Brute force matrix multiplication

int main(int argc, char *argv[]){


	if (MODE == 1){

		unsigned long long a, b;
		a = rdtsc();
		time_t t;
		int i,j,k;

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

		//multiply mats/////////////////////////
		for (i=0;i<SIZE;i++){
			for(j=0;j<SIZE;j++){
				for(k=0;k<SIZE;k++){
					matC[i][j] = matC[i][j] + (matA[i][k] * matB[k][j]);
					}
			}
		}


		//Display output/////////////////////////
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

		b = rdtsc();
		printf("\nDone. Number of clock Cycles: %llu\n", b-a);
	}

	else if (MODE == 2)
	{

		printf("size of long long is %d\n", sizeof(long long));
		printf("size of long is %d\n", sizeof(long));
		printf("size of int is %d\n", sizeof(int));
		printf("size of short is %d\n", sizeof(short));

		char *string = "test sting";
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
		printf("done\n");


	}
	else
	{

		printf("Invalid mode");

	}

	return 1;
}



void print_me(char *string)
{

	printf("Here is the string 2: \"%s\"\n", string);

}
