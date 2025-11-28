/*
 * timing_utils.h - CUDA timing helper utilities
 * 
 * Provides simple functions for timing CUDA kernel executions using CUDA events.
 * 
 * USAGE:
 *   #include "timing_utils.h"
 *   
 *   cudaEvent_t start, stop;
 *   create_cuda_timer(&start, &stop);
 *   
 *   start_cuda_timer(start);
 *   // ... your kernel launch ...
 *   float time_ms = stop_cuda_timer(start, stop);
 *   
 *   printf("Kernel took %.3f ms\n", time_ms);
 *   destroy_cuda_timer(start, stop);
 */

#ifndef TIMING_UTILS_H
#define TIMING_UTILS_H

#include <cuda_runtime.h>
#include <stdio.h>

/*
 * Create CUDA events for timing.
 */
inline void create_cuda_timer(cudaEvent_t *start, cudaEvent_t *stop) {
    cudaEventCreate(start);
    cudaEventCreate(stop);
}

/*
 * Start the timer by recording the start event.
 */
inline void start_cuda_timer(cudaEvent_t start) {
    cudaEventRecord(start, 0);
}

/*
 * Stop the timer and return elapsed time in milliseconds.
 */
inline float stop_cuda_timer(cudaEvent_t start, cudaEvent_t stop) {
    float elapsed_time;
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    return elapsed_time;
}

/*
 * Destroy CUDA events.
 */
inline void destroy_cuda_timer(cudaEvent_t start, cudaEvent_t stop) {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

/*
 * Convenience macro for timing a kernel.
 * 
 * EXAMPLE:
 *   TIME_KERNEL(
 *       my_kernel<<<blocks, threads>>>(args),
 *       "My Kernel"
 *   );
 */
#define TIME_KERNEL(kernel_call, name) \
    do { \
        cudaEvent_t _start, _stop; \
        create_cuda_timer(&_start, &_stop); \
        start_cuda_timer(_start); \
        kernel_call; \
        float _time = stop_cuda_timer(_start, _stop); \
        printf("%s took %.3f ms\n", name, _time); \
        destroy_cuda_timer(_start, _stop); \
    } while(0)

#endif // TIMING_UTILS_H
