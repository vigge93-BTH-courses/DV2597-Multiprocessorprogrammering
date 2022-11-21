/***************************************************************************
 *
 * Parallel version of Gaussian elimination
 * Don't recreate threads, possible deadlock
 ***************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include "pthread.h"
#include <math.h>
#include <string.h>

#define MAX_SIZE 4096
#define THREADS 8

typedef double matrix[MAX_SIZE][MAX_SIZE];
typedef struct eliminationArgs {
    int start;
    int end;
    int k;
    int n;
    int running;
} eliminationArgs;

int N;              /*matrix size */
int maxnum;         /*max number of element*/
char *Init;         /*matrix init type */
int PRINT;          /*print switch */
matrix A;           /*matrix A */
double b[MAX_SIZE]; /*vector b */
double y[MAX_SIZE]; /*vector y */
pthread_t threads[THREADS]; /* vector threads */
eliminationArgs eargs[THREADS]; /* vector divission thread arguments */
pthread_mutex_t mutexes_1[THREADS];
pthread_mutex_t mutexes_2[THREADS];
pthread_cond_t conds_1[THREADS];
pthread_cond_t conds_2[THREADS];


/*forward declarations */
void work(void);
void *eliminationWork(void *args);
void Init_Matrix(void);
void Print_Matrix(void);
void Init_Default(void);
int Read_Options(int, char **);

int main(int argc, char **argv)
{
    int i, timestart, timeend, iter;
    for (int j = 0; j < THREADS; j++) {
        pthread_mutex_init(&mutexes_1[j], NULL);
        pthread_mutex_init(&mutexes_2[j], NULL);
        pthread_cond_init(&conds_1[j], NULL);
        pthread_cond_init(&conds_2[j], NULL);
    }
    Init_Default();           /*Init default values */
    Read_Options(argc, argv); /*Read arguments */
    Init_Matrix();            /*Init the matrix */
    work();
    if (PRINT == 1)
        Print_Matrix();
}

void work(void)
{
    for (int n = 0; n < THREADS; n++) { // Start threads
        eargs[n].start = -1;
        eargs[n].end = -1;
        eargs[n].k = -1;
        eargs[n].n = n;
        eargs[n].running = 1;
        pthread_mutex_lock(&mutexes_2[n]);
        pthread_create(&threads[n], NULL, eliminationWork, &eargs[n]);
    }
    for (int i = 0; i < THREADS; i++) {
        pthread_mutex_lock(&mutexes_2[i]);
        pthread_mutex_unlock(&mutexes_2[i]);
    }
    for (int k = 0; k < N; k++)
    { /*Outer loop */
        double k_value = A[k][k];
        for (int j = k + 1; j < N; j++)
            A[k][j] = A[k][j] / k_value; /*Division step */
        y[k] = b[k] / k_value;
        A[k][k] = 1.0;
        int step = N / THREADS;
        int n = 0;
        for (int i = k + 1; i < N; i += step)
        {
            eargs[n].start = i;
            eargs[n].end = i + step;
            eargs[n].k = k;
            pthread_mutex_lock(&mutexes_1[n]);
            pthread_mutex_lock(&mutexes_2[n]);
            pthread_cond_signal(&conds_1[n]);
            pthread_mutex_unlock(&mutexes_1[n]);
            n++;
        }
        for (int j = n; j < THREADS; j++) {
            if (eargs[j].running == 1) {
                eargs[j].running = 0;
                pthread_mutex_lock(&mutexes_1[j]);
                pthread_cond_signal(&conds_1[j]);
                pthread_mutex_unlock(&mutexes_1[j]);
                pthread_join(threads[j], NULL);
            }
        }
        for (int j = 0; j < n; j++) {
            pthread_mutex_lock(&mutexes_2[j]);
            pthread_mutex_unlock(&mutexes_2[j]);
        }
    }
}

void *eliminationWork(void *args)
{
    eliminationArgs *data = (eliminationArgs*)args;
    int n = data->n;
    pthread_mutex_unlock(&mutexes_2[n]);
    while (data->running == 1) {
        // Wait for program to signal run
        pthread_cond_wait(&conds_1[n], &mutexes_1[n]);
        pthread_mutex_unlock(&mutexes_1[n]);
        if (data->running == 0) return NULL;
        int start = data->start;
        int end = data->end;
        int k = data->k;
        for (int i = start; i < end && i < N; i++) {
            for (int j = k + 1; j < N; j++)
                A[i][j] = A[i][j] - A[i][k] * A[k][j]; /*Elimination step */
            b[i] = b[i] - A[i][k] * y[k];
            A[i][k] = 0.0;
        }
        pthread_mutex_unlock(&mutexes_2[n]);
    }
}

void Init_Matrix()
{
    int i, j;
    printf("\nsize = %dx%d ", N, N);
    printf("\nmaxnum = %d \n", maxnum);
    printf("Init = %s \n", Init);
    printf("Initializing matrix...");
    if (strcmp(Init, "rand") == 0)
    {
        for (i = 0; i < N; i++)
        {
            for (j = 0; j < N; j++)
            {
                if (i == j) /*diagonal dominance */
                    A[i][j] = (double)(rand() % maxnum) + 5.0;
                else
                    A[i][j] = (double)(rand() % maxnum) + 1.0;
            }
        }
    }
    if (strcmp(Init, "fast") == 0)
    {
        for (i = 0; i < N; i++)
        {
            for (j = 0; j < N; j++)
            {
                if (i == j) /*diagonal dominance */
                    A[i][j] = 5.0;
                else
                    A[i][j] = 2.0;
            }
        }
    }
    /*Initialize vectors b and y */
    for (i = 0; i < N; i++)
    {
        b[i] = 2.0;
        y[i] = 1.0;
    }
    printf("done \n\n");
    if (PRINT == 1)
        Print_Matrix();
}

void Print_Matrix()
{
    int i, j;
    printf("Matrix A:\n");
    for (i = 0; i < N; i++)
    {
        printf("[");
        for (j = 0; j < N; j++)
            printf(" %5.2f,", A[i][j]);
        printf("]\n");
    }
    printf("Vector b:\n[");
    for (j = 0; j < N; j++)
        printf(" %5.2f,", b[j]);
    printf("]\n");
    printf("Vector y:\n[");
    for (j = 0; j < N; j++)
        printf(" %5.2f,", y[j]);
    printf("]\n");
    printf("\n\n");
}

void Init_Default()
{
    N = 2048;
    Init = "rand";
    maxnum = 15.0;
    PRINT = 0;
}

int Read_Options(int argc, char **argv)
{
    char *prog;
    prog = *argv;
    while (++argv, --argc > 0)
        if (**argv == '-')
            switch (*++*argv)
            {
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
                printf(" [-D] show default values \n");
                printf(" [-h] help \n");
                printf(" [-I init_type] fast/rand \n");
                printf(" [-m maxnum] max random no \n");
                printf(" [-P print_switch] 0/1 \n");
                exit(0);
                break;
            case 'D':
                printf("\nDefault: n = %d ", N);
                printf("\n Init = rand");
                printf("\n maxnum = 15 ");
                printf("\n P = 0 \n\n");
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