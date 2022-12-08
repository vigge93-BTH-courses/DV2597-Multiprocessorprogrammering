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
    // printf("Gauss Jordan\n");

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
int getIndex(int y, int x, int N) {
    return x + y*N;
}

__global__
void divisionKernel(double *A_d, double *b_d, double *y_d, int N, int k) {
    int tIdX = threadIdx.x + blockDim.x*blockIdx.x + k+1; // Start thread index at k+1 since all indexes <= k is already eliminated.
    y_d[k] = b_d[k] / A_d[getIndex(k, k, N)]; // Idempotent operation within a kernel launch.
    if (tIdX >= N) return; // Guard statement
    A_d[getIndex(k, tIdX, N)] = A_d[getIndex(k, tIdX, N)] / A_d[getIndex(k, k, N)];; /* Division step */
}

__global__
void eliminationKernel(double *A_d, double *b_d, double *y_d, int N, int k) {
    int tIdX = threadIdx.x + blockDim.x*blockIdx.x; // column
    int tIdY = threadIdx.y + blockDim.y*blockIdx.y; // row
    A_d[getIndex(k, k, N)] = 1.0; // We set A[k][k] to 1 here in order to avoid race condition in division kernel. Idempotent operation
    
    if (tIdY == k || tIdY >= N) return; // Guard statement

    if (tIdX == 0) { // Run exactly once per iteration.
        double* ptr = tIdY > k ? &b_d[tIdY] : &y_d[tIdY];
        *ptr = *ptr - A_d[getIndex(tIdY, k, N)] * y_d[k];
    }

    if (tIdX <= k || tIdX >= N) return; // Guard statement
    A_d[getIndex(tIdY, tIdX, N)] = A_d[getIndex(tIdY, tIdX, N)] - A_d[getIndex(tIdY, k, N)] * A_d[getIndex(k, tIdX, N)]; /* Elimination step */
}

void
work(void)
{
    int threadsPerDivideBlock = 256;
    dim3 threadsPerEliminationBlock(16, 16, 1);
    int noDivideBlocks = ceil(N/(float)threadsPerDivideBlock);
    dim3 eliminationBlocks(ceil(N/(float)threadsPerEliminationBlock.x), ceil(N/(float)threadsPerEliminationBlock.y), 1);
    
    double *A_d;
    double *b_d, *y_d;
    
    cudaMalloc((void**)&A_d,N*N*sizeof(double));
    cudaMalloc((void**)&b_d,N*sizeof(double));
    cudaMalloc((void**)&y_d,N*sizeof(double));
    
    for (int i = 0; i < N; i++) {
        cudaMemcpy(&A_d[i*N], A[i], N*sizeof(double), cudaMemcpyHostToDevice);
    }
    cudaMemcpy(b_d, b, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(y_d, y, N*sizeof(double), cudaMemcpyHostToDevice);
    
    for (int k = 0; k < N; k++) {
        divisionKernel<<<noDivideBlocks, threadsPerDivideBlock>>>(A_d, b_d, y_d, N, k);
        eliminationKernel<<<eliminationBlocks, threadsPerEliminationBlock>>>(A_d, b_d, y_d, N, k);
    }
    
    cudaMemcpy(y, y_d, N*sizeof(double), cudaMemcpyDeviceToHost); // We only care about the result vector
    
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

    // printf("Matrix A:\n");
    // for (i = 0; i < N; i++) {
    //     printf("[");
    //     for (j = 0; j < N; j++)
    //         printf(" %5.2f,", A[i][j]);
    //     printf("]\n");
    // }
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