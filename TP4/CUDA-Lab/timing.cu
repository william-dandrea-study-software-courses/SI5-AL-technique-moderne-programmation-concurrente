/* Same as 03, but use a Unified memory */

#include <iostream>
#include <time.h>
#include <sys/time.h>
#include <stdint.h>
#include <math.h>

uint64_t get_posix_clock_time ()
{
    struct timeval tv;

    if (gettimeofday (&tv, NULL) == 0)
        return (uint64_t) (tv.tv_sec * 1000000 + tv.tv_usec)/1000;
    else
        return 0;

}

void print_duration(char *txt, uint64_t  start, uint64_t stop)
{
    std::cout << txt << ": " << stop-start << "ms" << std::endl;
}

float check_error(float *z, int N)
{
  // Check for errors (all values should be 3.0f)
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(z[i]-3.0f));
  return maxError;
}

// function to add the elements of two arrays
void add_CPU(int n, float *x, float *y, float *z)
{
  for (int i = 0; i < n; i++)
      z[i] = x[i] + y[i];
}

__global__
void add_GPU(int n, float *x, float *y, float *z)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
      z[i] = x[i] + y[i];
}

int main(void)
{
  int N = 1<<20;            // 1M elements
  float *x, *y, *z;         // data on unified memory

  float maxError;           // to count error
  uint64_t start, stop;     // to measure execution time on CPU
  cudaEvent_t d_start, d_stop;// to measure execution time on GPU
  float time;

  // Allocate data on unified memory
  cudaMallocManaged(&x, N*sizeof(float));
  cudaMallocManaged(&y, N*sizeof(float));
  cudaMallocManaged(&z, N*sizeof(float));

  cudaEventCreate (&d_start);
  cudaEventCreate (&d_stop);

  
  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  // Run kernel on 1M elements on the device in 1 block and 1 thread
  {
    start = get_posix_clock_time();

    cudaEventRecord (d_start, 0); 

    add_GPU<<<1, 1>>>(N, x, y, z);
    
    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();
    cudaEventRecord (d_stop, 0);
    cudaEventSynchronize (d_stop);

    stop = get_posix_clock_time();
    
    // Check for errors (all values should be 3.0f)
    if ((maxError=check_error(z, N))==0) {
       print_duration((char *) "1 thread on device", start, stop);
       cudaEventElapsedTime (&time, d_start, d_stop);
       std::cout << "Elapsed time on GPU: " << time << "ms\n";
    } else std::cout << "Max error: " << maxError << std::endl;
   }

  // Free memory
  cudaFree(x);
  cudaFree(y);
  cudaFree(z);

return 0;
}
