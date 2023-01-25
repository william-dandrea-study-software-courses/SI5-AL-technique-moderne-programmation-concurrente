#ifndef _DEFINITIONS_H_
#define _DEFINITIONS_H_

#include <malloc.h>
#include <math.h>
#include <sys/timeb.h>

#ifndef N_THREADS
#   define N_THREADS 1000
#endif

#ifndef USE_CPUS
#   define USE_CPUS 1
#endif

#define CACHE_LINE_SIZE            64

#define MAX_WORK                   64
#define SIM_RAND_MAX               32767
#define RUNS                       (10000000 / N_THREADS)
#define CACHE_ALIGN                __attribute__ ((aligned (CACHE_LINE_SIZE)))

#define Object                     int32_t
#define null                       NULL
#define bool                       int32_t
#define true                       1
#define false                      0

inline static void *getAlignedMemory(size_t align, size_t size) {
    void *p;

    p = (void *)memalign(align, size);
    if (p == null) {
        perror("memalign");
        exit(EXIT_FAILURE);
    } else return p;
}

inline static int64_t getTimeMillis(void) {
    struct timeval tm;

    gettimeofday(&tm, NULL);
    return ((tm.tv_sec * 1000000) + tm.tv_usec)/1000;
}

int _thread_pin(unsigned int cpu_id)
{
    pthread_setconcurrency(USE_CPUS);

    int ret = 0;
    cpu_set_t mask;
    unsigned int len = sizeof(mask);

    CPU_ZERO(&mask);
    CPU_SET(cpu_id % USE_CPUS, &mask);
    ret = sched_setaffinity(0, len, &mask);
    if (ret == -1)
        perror("sched_setaffinity");

    return ret;
}

__thread int32_t __thread_id = -1;

inline static void setThreadId(int32_t id) {
    __thread_id = id;
}

inline static int32_t getThreadId(void) {
    return __thread_id;
}

__thread unsigned long next = 1;

inline static long simRandom(void) {
    next = next * 1103515245 + 12345;
    return((unsigned)(next/65536) % 32768);
}

inline static void simSRandom(unsigned long seed) {
    next = seed;
}

inline static long simRandomRange(long low, long high) {
    return low + (long)(((double) high)* (simRandom() / (SIM_RAND_MAX + 1.0)));
}

#endif

