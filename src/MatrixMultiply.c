#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "rdtsc.h"


#define SIZE 10
#define MODE 1

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

	else {
		printf("select another mode\n");
	}

	return 1;
}
