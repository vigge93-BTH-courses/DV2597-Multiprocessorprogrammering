/***************************************************************************
 *
 *Sequential version of Matrix-Matrix multiplication
 *
 ***************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#define SIZE 2048

static double a[SIZE][SIZE];
static double b[SIZE][SIZE];
static double c[SIZE][SIZE];

struct threadArgs
{
    unsigned int row;
};

pthread_t *threads;
struct threadArgs *args;

static void
init_matrix(void)
{
    int i, j;
    for (i = 0; i < SIZE; i++)
        for (j = 0; j < SIZE; j++)
        {
            /*Simple initialization, which enables us to easy check
             *the correct answer. Each element in c will have the same
             *value as SIZE after the matmul operation.
             */
            a[i][j] = 1.0;
            b[i][j] = 1.0;
        }
}

void *mul_row(void *params)
{
    struct threadArgs *args = (struct threadArgs *)params;
    unsigned int row = args->row;
    for (int i = row; i < row + (SIZE / 8); i++)
    {
        for (int j = 0; j < SIZE; j++)
        {
            c[i][j] = 0.0;
            for (int k = 0; k < SIZE; k++)
                c[i][j] = c[i][j] + a[i][k] * b[k][j];
        }
    }
}

static void
matmul_seq()
{
    int i, j, k;
    for (i = 0; i < 8; i++)
    {
        args[i].row = i * (SIZE / 8);
        pthread_create(&threads[i], NULL, mul_row, &args[i]);
    }
    for (i = 0; i < 8; i++)
    {
        pthread_join(threads[i], NULL);
    }
}

static void
print_matrix(void)
{
    int i, j;
    for (i = 0; i < SIZE; i++)
    {
        for (j = 0; j < SIZE; j++)
            printf(" %7.2f", c[i][j]);
        printf("\n");
    }
}

int main(int argc, char **argv)
{
    threads = malloc(8 * sizeof(pthread_t));
    args = malloc(8 * sizeof(struct threadArgs));
    init_matrix();
    matmul_seq();
    free(threads);
    free(args);
    // print_matrix();
}