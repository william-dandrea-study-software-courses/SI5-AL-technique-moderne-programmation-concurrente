#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <omp.h>
#include <string.h>
#include <stdint.h>
#include <immintrin.h>

#include "definitions.h"


/******************************************************************************/
/* The lock                                                                   */
/******************************************************************************/
// pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
// bool lock = false;
/******************************************************************************/

typedef struct ListNode {
    Object value; 
    volatile struct ListNode *next;
} ListNode;	

int counter = 0;
ListNode guard CACHE_ALIGN = {0, null};
volatile ListNode *Head CACHE_ALIGN = &guard;
int64_t d1 CACHE_ALIGN, d2;



void ticketLock_init(int *next_ticket, int *now_serving) {
    *now_serving = *next_ticket = 0;
}

void ticketLock_acquire(int *next_ticket, int *now_serving) {
    int *my_ticket = __sync_fetch_and_add(&next_ticket, 1);
    while (*now_serving != *my_ticket) {
        _mm_pause();
        // sched_yield();
    }
}

void ticketLock_release(int *now_serving) {
    __sync_fetch_and_add(&now_serving, 1);
}


int now_serving = 0;
int next_ticket = 0;

inline static void push(Object arg, int pid) {
    volatile ListNode *n = getAlignedMemory(CACHE_LINE_SIZE, sizeof(ListNode));
    n->value = (Object)arg;

    /**************************************************************************/
    /* Critical section                                                       */
    /**************************************************************************/
    ticketLock_acquire(&next_ticket, &now_serving);

    n->next = Head;
    Head = n;

    ticketLock_release(&now_serving);
    /**************************************************************************/
}

inline static Object pop(int pid) {
    Object result;

    /**************************************************************************/
    /* Critical section                                                       */
    /**************************************************************************/
    ticketLock_acquire(&next_ticket, &now_serving);

    if (Head->next == null) 
        result = -1;
    else {
        result = Head->next->value;
        Head = Head->next;
    }

    ticketLock_release(&now_serving);
    
    /**************************************************************************/

    return result;
}

pthread_barrier_t barr;


inline static void Execute(void* Arg) {
    long i;
    long rnum;
    long id = (long) Arg;
    volatile int j;

    setThreadId(id);
    _thread_pin(id);
    simSRandom(id + 1);
    
    if (id == N_THREADS - 1)
        d1 = getTimeMillis();
    
    int rc = pthread_barrier_wait(&barr);
    
    if (rc != 0 && rc != PTHREAD_BARRIER_SERIAL_THREAD) {
        printf("Could not wait on barrier\n");
        exit(-1);
    }
    
    for (i = 0; i < RUNS; i++) {
        push((Object)1, id);
        rnum = simRandomRange(1, MAX_WORK);
        for (j = 0; j < rnum; j++)
            ;
        pop(id);
        rnum = simRandomRange(1, MAX_WORK);
        for (j = 0; j < rnum; j++)
            ;
    }
}

inline static void* EntryPoint(void* Arg) {
    Execute(Arg);
    return null;
}
    

inline static pthread_t StartThread(int arg) {
    long id = (long) arg;
    void *Arg = (void*) id;
    pthread_t thread_p;

    pthread_attr_t my_attr;
    pthread_attr_init(&my_attr);
    pthread_create(&thread_p, &my_attr, EntryPoint, Arg);

    return thread_p;
}

int main(void) {
    pthread_t threads[N_THREADS];
    int i;

    // ticketLock_init(&next_ticket, &now_serving);

    if (pthread_barrier_init(&barr, NULL, N_THREADS)) {
        printf("Could not create the barrier\n");
        return -1;
    }

    for (i = 0; i < N_THREADS; i++)
        threads[i] = StartThread(i);

    for (i = 0; i < N_THREADS; i++)
        pthread_join(threads[i], NULL);
    d2 = getTimeMillis();

    printf("time: %d ns\n", (int) (d2 - d1));
    
    if (pthread_barrier_destroy(&barr)) {
        printf("Could not destroy the barrier\n");
        return -1;
    }

    return 0;
}

