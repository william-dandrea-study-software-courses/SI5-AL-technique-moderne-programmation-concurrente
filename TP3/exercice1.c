#include <stdio.h>
#include <stdint.h>
#include <pthread.h>
#include <sys/time.h>

#define THREADS 4
#define ITERATIONS 100000000

static __thread unsigned long x, y=362436069, z=521288629;

uint64_t results[THREADS];

unsigned long xorshf96(void) {
    unsigned long t;

    x ^= x << 16;
    x ^= x >> 5;
    x ^= x << 1;

    t = x;
    x = y;
    y = z;
    z = t ^ x ^ y;

    return z;
}

void* execution(void* thread_id) {
    int typed_thread_id = (int)(intptr_t)thread_id;


    pthread_t current_thread = pthread_self();
    pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &typed_thread_id);

    struct timeval now;
    gettimeofday(&now, NULL);
    x = now.tv_usec + typed_thread_id;

    for (int i = 0; i < ITERATIONS; i++) {
        results[typed_thread_id] += xorshf96() % 2;
    }

    return NULL;
}

struct timeval start, end;

int main(int argc, char *argv[]) {

    pthread_t threads[THREADS];
    gettimeofday(&start, NULL);

    for (int i = 0; i < THREADS; i++) {
        pthread_create(&threads[i], NULL, execution, (void*)(intptr_t)i);
    }

    for (int i = 0; i < THREADS; i++) {
        pthread_join(threads[i], NULL);
    }

    for (int i = 0; i < THREADS; i++) {
        printf("results[%d] = %llu \n", i, results[i]);
    }
    gettimeofday(&end, NULL);

    printf ("Execution time : %f seconds\n", (double) (end.tv_usec - start.tv_usec) / 1000000 + (double) (end.tv_sec - start.tv_sec));

    return 0;
}