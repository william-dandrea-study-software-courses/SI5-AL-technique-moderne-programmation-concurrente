/**
 * Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 * Some modification for UCA Lab on october 2021
 */

/**
 * Vector addition: C = A + B.
 *
 * This sample is a very basic sample that implements element by element
 * vector addition. It is the same as the sample illustrating Chapter 2
 * of the programming guide with some additions like error checking.
 */

#include <stdio.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
#include <time.h>
#include <sys/time.h>

/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. 
 * The 3 vectors have the same number of elements numElements.
 */
__global__ void vectorAdd(const float *A, const float *B, float *C, unsigned long numElements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements) {
        C[i] = A[i] + B[i];
    }
}

void checkErr(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "%s (error code %d: '%s')!\n", msg, err, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

float timedifference_msec(struct timeval t0, struct timeval t1) {
    return (t1.tv_sec - t0.tv_sec) * 1000.0f + (t1.tv_usec - t0.tv_usec) / 1000.0f;
}


/**
 * Host main routine
 */
int main(int argc, char **argv) {
    // Get the property device
    // TO BE COMPLETED
    int threadsPerBlock = 1024;
    int maxBlocks = 10865535;
    printf("max of %d blocks of %d threads\n", maxBlocks, threadsPerBlock);

    // To mesure different execution time
    // TO BE COMPLETED

    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    // Print the vector length to be used, and compute its size
    unsigned long numElements = 50000;
    if (argc == 2) {
        numElements = strtoul(argv[1], 0, 10);
    }
    size_t size = numElements * sizeof(float);
    printf("[Vector addition of %lu elements]\n", numElements);

    // Allocate the host input vectors A & B
    float *h_A = (float *) malloc(size);
    float *h_B = (float *) malloc(size);

    // Allocate the host output vector C
    float *h_C = (float *) malloc(size);

    // Verify that allocations succeeded
    if (h_A == NULL || h_B == NULL || h_C == NULL) {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    // Initialize the host input vectors
    for (int i = 0; i < numElements; ++i) {
        h_A[i] = rand() / (float) RAND_MAX;
        h_B[i] = rand() / (float) RAND_MAX;
    }

    // 1a. Allocate the device input vectors A & B
    float *d_A = NULL;
    err = cudaMalloc((void **) &d_A, size);
    checkErr(err, "Failed to allocate device vector A");
    float *d_B = NULL;
    err = cudaMalloc((void **) &d_B, size);
    checkErr(err, "Failed to allocate device vector B");

    // 1.b. Allocate the device output vector C
    float *d_C = NULL;
    err = cudaMalloc((void **) &d_C, size);
    checkErr(err, "Failed to allocate device vector C");

    // 2. Copy the host input vectors A and B in host memory 
    //     to the device input vectors in device memory
    printf("Copy input data from the host memory to the CUDA device\n");
    struct timeval endCopy, startCopy;
    cudaError_t err1 = cudaSuccess;
    cudaError_t err2 = cudaSuccess;
    gettimeofday(&startCopy, 0);
    err1 = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    err2 = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    gettimeofday(&endCopy, 0);

    checkErr(err1, "Failed to copy device vector A from host to device");
    checkErr(err2, "Failed to copy device vector B from host to device");
    float CUDA1 = (float) timedifference_msec(startCopy, endCopy);
    printf("CUDA copying time from host to device: %lf ms\n", CUDA1);

    // 3. Launch the Vector Add CUDA Kernel
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    if (blocksPerGrid > maxBlocks) {
        fprintf(stderr, "too much blocks %d!\n", blocksPerGrid);
        exit(EXIT_FAILURE);
    } else
        printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

    struct timeval endAdd, startAdd;
    gettimeofday(&startAdd, 0);
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
    gettimeofday(&endAdd, 0);
    err = cudaGetLastError();
    checkErr(err, "Failed to launch vectorA:dd kernel");


    float CUDA2 = timedifference_msec(startAdd, endAdd);
    printf("time spent computing the sum of the A and B arrays into C: %lf ms\n", CUDA2);

    // 4. Copy the device result vector in device memory
    //     to the host result vector in host memory.
    printf("Copy output data from the CUDA device to the host memory\n");
    struct timeval endCopy2, startCopy2;
    gettimeofday(&startCopy2, 0);
    err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    gettimeofday(&endCopy2, 0);
    checkErr(err, "Failed to copy vector C from device to host");


    float CUDA3 = timedifference_msec(startCopy2, endCopy2);
    printf("CUDA copying time from device to host: %lf\n", CUDA3);

    // Verify that the result vector is correct
    for (int i = 0; i < numElements; ++i) {
        if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5) {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }
    printf("CUDA test PASSED\n");
    float cudaTime = CUDA1 + CUDA2 + CUDA3;
    printf("CUDA time: %lf\n", cudaTime);

    // Free device global memory
    err = cudaFree(d_A);
    checkErr(err, "Failed to free device vector A");

    err = cudaFree(d_B);
    checkErr(err, "Failed to free device vector B");

    err = cudaFree(d_C);
    checkErr(err, "Failed to free device vector C");

    // repeat the computation sequentially
    struct timeval endSeq, startSeq;
    gettimeofday(&startSeq, 0);
    for (int i = 0; i < numElements; ++i) {
        h_C[i] = h_A[i] + h_B[i];
    }
    gettimeofday(&endSeq, 0);

    // verify again
    for (int i = 0; i < numElements; ++i) {
        if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5) {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }
    printf("\nNormal test PASSED\n");

    float SEQ = timedifference_msec(startSeq, endSeq);;
    printf("Normal time: %lf\n", SEQ);

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    // Reset the device and exit
    err = cudaDeviceReset();
    checkErr(err, "Unable to reset device");

    fprintf(stderr, "%d, %d, %d, %lf, %lf, %lf, %lf, %lf \n", numElements, blocksPerGrid, threadsPerBlock, CUDA1, CUDA2, CUDA3, cudaTime, SEQ);

    printf("Done\n");
    return 0;
}

