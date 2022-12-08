/***************************************************************************
 *
 * Paralell version of Gauss-Jordan row reduction
 *
 ***************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <chrono>
#include <cuda.h>

#define MAX_SIZE 4096

typedef double matrix[MAX_SIZE][MAX_SIZE];

int	N;		/* matrix size */
int	maxnum;		/* max number of element*/
char* Init;		/* matrix init type	*/
int	PRINT;		/* print switch */
matrix	A;		/* matrix A	*/
double	b[MAX_SIZE];	/* vector b */
double	y[MAX_SIZE];	/* vector y */

/* forward declarations */
void work(void);
void Init_Matrix(void);
void Print_Matrix(void);
void Init_Default(void);
int Read_Options(int, char**);

int
main(int argc, char** argv)
{
    srand(1);
    // printf("Gauss Jordan\n");
    // int i, timestart, timeend, iter;

    Init_Default();		/* Init default values	*/
    Read_Options(argc, argv);	/* Read arguments	*/
    Init_Matrix();		/* Init the matrix	*/
    auto start = std::chrono::steady_clock::now();
    work();
    auto end = std::chrono::steady_clock::now();
    std::cout << "Elapsed time =  " << std::chrono::duration<double>(end - start).count() << " sec\n";
    if (PRINT == 1)
        Print_Matrix();
}

__device__
int getIndex(int y, int x) {
    return x + y*MAX_SIZE;
}

__global__
void division_kernel(double *A_d, double *b_d, double *y_d, int N, int k, int stride) {
    int t_idx = threadIdx.x + blockDim.x*blockIdx.x + k; // start thread index at k since all indexes < k is already eliminated.
    if (t_idx >= N) return; // Guard clause
    double A_kk = A_d[getIndex(k, k)];
    for (int j = t_idx + 1; j < N; j += stride) {
        A_d[getIndex(k, j)] = A_d[getIndex(k, j)] / A_kk; /* Division step */
    }
    if (t_idx == k) { // Only execute once per kernel launch
        y_d[k] = b_d[k] / A_kk;
        A_d[getIndex(k, k)] = 1.0; // This causes race condition
    }
}

__global__
void elimination_kernel(double *A_d, double *b_d, double *y_d, int N, int k, int stride) {
    int t_idx = threadIdx.x + blockDim.x*blockIdx.x + k;
    if (t_idx >= N) return; // Guard clause
    for (int i = t_idx + 1; i < N; i += stride) {
        for (int j = k + 1; j < N; j++) {
            A_d[getIndex(i, j)] = A_d[getIndex(i, j)] - A_d[getIndex(i, k)] * A_d[getIndex(k, j)]; /* Elimination step */
        }
        b_d[i] = b_d[i] - A_d[getIndex(i, k)] * y_d[k];
        A_d[getIndex(i, k)] = 0.0;
    }
}

__global__
void jordan_kernel(double *A_d, double *b_d, double *y_d, int N, int k, int stride) {
    int t_idx = threadIdx.x + blockDim.x*blockIdx.x;
    if (t_idx >= k) return; // Guard clause
    for (int i = t_idx; i < k; i += stride) {
        for (int j = k + 1; j < N; j++) {
            A_d[getIndex(i, j)] = A_d[getIndex(i, j)] - A_d[getIndex(i, k)] * A_d[getIndex(k, j)]; /* Additional Elimination for Gauss-Jordan */
        }
        y_d[i] = y_d[i] - A_d[getIndex(i, k)] * y_d[k];
        A_d[getIndex(i, k)] = 0.0;
    }
}

void
work(void)
{
    /* Gaussian elimination algorithm, Algo 8.4 from Grama */
    int blocks = 1;
    int threads_per_block = 1024;
    int stride = blocks*threads_per_block;
    double *A_d;
    double *b_d, *y_d;
    cudaMalloc((void**)&A_d,MAX_SIZE*MAX_SIZE*sizeof(double));
    cudaMalloc((void**)&b_d,MAX_SIZE*sizeof(double));
    cudaMalloc((void**)&y_d,MAX_SIZE*sizeof(double));
    for (int i = 0; i < MAX_SIZE; i++) {
        cudaMemcpy(&A_d[i*MAX_SIZE], A[i], MAX_SIZE*sizeof(double), cudaMemcpyHostToDevice);
    }
    cudaMemcpy(b_d, b, MAX_SIZE*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(y_d, y, MAX_SIZE*sizeof(double), cudaMemcpyHostToDevice);
    for (int k = 0; k < N; k++) { /* Outer loop */
        division_kernel<<<blocks, threads_per_block>>>(A_d, b_d, y_d, N, k, stride);
        // printf("Error: %s\n", cudaGetErrorString(cudaGetLastError()));

        elimination_kernel<<<blocks, threads_per_block>>>(A_d, b_d, y_d, N, k, stride);
        // printf("Error: %s\n", cudaGetErrorString(cudaGetLastError()));

        jordan_kernel<<<blocks, threads_per_block>>>(A_d, b_d, y_d, N, k, stride);
        // printf("Error: %s\n", cudaGetErrorString(cudaGetLastError()));
    }
    // for (int i = 0; i < MAX_SIZE; i++) {
    //     cudaMemcpy(A[i], &A_d[i*MAX_SIZE], MAX_SIZE*sizeof(double), cudaMemcpyDeviceToHost);
    // }
    cudaMemcpy(b, b_d, MAX_SIZE*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(y, y_d, MAX_SIZE*sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(A_d);
    cudaFree(b_d);
    cudaFree(y_d);
}

void
Init_Matrix()
{
    int i, j;

    // printf("\nsize      = %dx%d ", N, N);
    // printf("\nmaxnum    = %d \n", maxnum);
    // printf("Init	  = %s \n", Init);
    // printf("Initializing matrix...");

    if (strcmp(Init, "rand") == 0) {
        for (i = 0; i < N; i++) {
            for (j = 0; j < N; j++) {
                if (i == j) /* diagonal dominance */
                    A[i][j] = (double)(rand() % maxnum) + 5.0;
                else
                    A[i][j] = (double)(rand() % maxnum) + 1.0;
            }
        }
    }
    if (strcmp(Init, "fast") == 0) {
        for (i = 0; i < N; i++) {
            for (j = 0; j < N; j++) {
                if (i == j) /* diagonal dominance */
                    A[i][j] = 5.0;
                else
                    A[i][j] = 2.0;
            }
        }
    }

    /* Initialize vectors b and y */
    for (i = 0; i < N; i++) {
        b[i] = 2.0;
        y[i] = 1.0;
    }

    // printf("done \n\n");
    if (PRINT == 1)
        Print_Matrix();
}

void
Print_Matrix()
{
    int i, j;

    printf("Matrix A:\n");
    for (i = 0; i < N; i++) {
        printf("[");
        for (j = 0; j < N; j++)
            printf(" %5.2f,", A[i][j]);
        printf("]\n");
    }
    printf("Vector y:\n[");
    for (j = 0; j < N; j++)
        printf(" %5.2f,", y[j]);
    printf("]\n");
    printf("\n\n");
}

void
Init_Default()
{
    N = 2048;
    Init = "fast";
    maxnum = 15.0;
    PRINT = 0;
}

int
Read_Options(int argc, char** argv)
{
    char* prog;

    prog = *argv;
    while (++argv, --argc > 0)
        if (**argv == '-')
            switch (*++ * argv) {
            case 'n':
                --argc;
                N = atoi(*++argv);
                break;
            case 'h':
                printf("\nHELP: try sor -u \n\n");
                exit(0);
                break;
            case 'u':
                printf("\nUsage: gaussian [-n problemsize]\n");
                printf("           [-D] show default values \n");
                printf("           [-h] help \n");
                printf("           [-I init_type] fast/rand \n");
                printf("           [-m maxnum] max random no \n");
                printf("           [-P print_switch] 0/1 \n");
                exit(0);
                break;
            case 'D':
                printf("\nDefault:  n         = %d ", N);
                printf("\n          Init      = rand");
                printf("\n          maxnum    = 5 ");
                printf("\n          P         = 0 \n\n");
                exit(0);
                break;
            case 'I':
                --argc;
                Init = *++argv;
                break;
            case 'm':
                --argc;
                maxnum = atoi(*++argv);
                break;
            case 'P':
                --argc;
                PRINT = atoi(*++argv);
                break;
            default:
                printf("%s: ignored option: -%s\n", prog, *argv);
                printf("HELP: try %s -u \n\n", prog);
                break;
            }
}