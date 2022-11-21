/***************************************************************************
 *
 * Parallel version of Quick sort until a certain block size
 *
 ***************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <signal.h>
#define KILO (1024)
#define MEGA (1024 * 1024)
#define MAX_ITEMS (64 * MEGA)
#define SMALLEST_ARRAY (MAX_ITEMS/(1024)) // Minimum block size
#define swap(v, a, b) \
    {                 \
        unsigned tmp; \
        tmp = v[a];   \
        v[a] = v[b];  \
        v[b] = tmp;   \
    }

typedef struct ThreadArgs
{
    int *v;
    unsigned int low;
    unsigned int high;
} ThreadArgs;

static int *v;

static void
print_array(void)
{
    int i;
    for (i = 0; i < MAX_ITEMS; i++)
        printf("%d ", v[i]);
    printf("\n");
}

static void
init_array(void)
{
    int i;
    v = (int *)malloc(MAX_ITEMS * sizeof(int));
    for (i = 0; i < MAX_ITEMS; i++)
        v[i] = rand();
}

static unsigned
partition(int *v, unsigned low, unsigned high, unsigned pivot_index)
{
    /*move pivot to the bottom of the vector */
    if (pivot_index != low)
        swap(v, low, pivot_index);
    pivot_index = low;
    low++;
    /*invariant:
     *v[i] for i less than low are less than or equal to pivot
     *v[i] for i greater than high are greater than pivot
     */
    /*move elements into place */
    while (low <= high)
    {
        if (v[low] <= v[pivot_index])
            low++;
        else if (v[high] > v[pivot_index])
            high--;
        else
            swap(v, low, high);
    }
    /*put pivot back between two groups */
    if (high != pivot_index)
        swap(v, pivot_index, high);
    return high;
}

void *
quick_sort(void *params)
{
    ThreadArgs *tparams = (ThreadArgs *)params;
    int *v = tparams->v;
    unsigned int low = tparams->low;
    unsigned int high = tparams->high;
    unsigned pivot_index;
    /*no need to sort a vector of zero or one element */
    if (low >= high)
        return 0;
    /*select the pivot value */
    pivot_index = (low + high) / 2;
    /*partition the vector */
    pivot_index = partition(v, low, high, pivot_index);
    ThreadArgs *args = malloc(sizeof(ThreadArgs)*2);
    /*sort the two sub arrays */
    pthread_t localThreads[2] = {-1, -1};
    if (low < pivot_index)
    {
        args[0].v = v;
        args[0].low = low;
        args[0].high = pivot_index - 1;
        if (pivot_index - low >= SMALLEST_ARRAY)
        {  // Run in parallel
            pthread_create(&localThreads[0], NULL, quick_sort, &args[0]);
        }
    }
    if (pivot_index < high)
    {
        args[1].v = v;
        args[1].low = pivot_index + 1;
        args[1].high = high;
        if (high - pivot_index < SMALLEST_ARRAY)
        { // Run in sequential
            quick_sort(&args[1]);
        }
        else
        { // Run in parallel
            pthread_create(&localThreads[1], NULL, quick_sort, &args[1]);
        }
    }
    if (low < pivot_index && localThreads[0] == -1) quick_sort(&args[0]); // Run low partition in sequential
    for (int i = 0; i < 2; i++) {
        if (localThreads[i] != -1) {
            pthread_join(localThreads[i], NULL);
        }
    }
    free(args);
}

int main(int argc, char **argv)
{
    init_array();
    ThreadArgs args;
    args.v = v;
    args.low = 0;
    args.high = MAX_ITEMS - 1;
    quick_sort(&args);
}