/*
 * direct_cuda.cu - CUDA C implementation skeleton for direct summation
 * 
 * TODO: Implement the direct summation method for calculating gravitational potential.
 * 
 * The potential at position i is:
 *   Φ_i = -G * Σ(j≠i) m_j / |r_i - r_j|
 * 
 * INSTRUCTIONS:
 * 1. Complete the compute_potential_kernel function
 * 2. Each thread should compute the potential for one particle
 * 3. Use a softening parameter (SOFTENING) to avoid singularities
 * 4. The host function is provided - you only need to implement the kernel
 * 
 * COMPILATION:
 *   nvcc -o direct_cuda direct_cuda.cu
 * 
 * HINTS:
 * - Use blockIdx.x and threadIdx.x to get the particle index
 * - Loop over all other particles to sum contributions
 * - Use sqrtf() for square root calculation
 * - Don't forget to check if i != j to avoid self-interaction
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cuda_runtime.h>
#include "../utils/json_writer.h"

#define SOFTENING 0.01f  // Softening parameter to avoid singularities (ε=0.01)

/*
 * CUDA kernel for direct summation of gravitational potential.
 * 
 * TODO: Implement this kernel
 * 
 * Parameters:
 *   x, y, z     - Arrays of particle positions (device memory)
 *   masses      - Array of particle masses (device memory)
 *   potentials  - Output array for potentials (device memory)
 *   n_bodies    - Total number of particles
 *   G           - Gravitational constant
 */
__global__ void compute_potential_kernel(
    const float *x, const float *y, const float *z,
    const float *masses,
    float *potentials,
    int n_bodies,
    float G)
{
    // Get the index of the current particle
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n_bodies) {
        // Initialize potential to zero
        float pot = 0.0f;

        // Get the position of particle i
        float xi = x[i];
        float yi = y[i];
        float zi = z[i];

        // Precompute softening squared
        float softening_sq = SOFTENING * SOFTENING;

        // Loop over all particles j and sum contributions
        for (int j = 0; j < n_bodies; j++) {
            // Skip self-interaction
            if (i == j) continue;

            // Calculate displacement
            float dx = xi - x[j];
            float dy = yi - y[j];
            float dz = zi - z[j];

            // Calculate distance with softening
            float r = sqrtf(dx*dx + dy*dy + dz*dz + softening_sq);

            // Add contribution: Φ_i -= G * m_j / r
            pot -= G * masses[j] / r;
        }

        // Store the result
        potentials[i] = pot;
    }
}

/*
 * Host function to compute potentials using direct summation.
 * This function is complete - no changes needed here.
 */
extern "C" void compute_potential_direct(
    const float *h_x, const float *h_y, const float *h_z,
    const float *h_masses,
    float *h_potentials,
    int n_bodies,
    float G)
{
    // Allocate device memory
    float *d_x, *d_y, *d_z, *d_masses, *d_potentials;
    size_t size = n_bodies * sizeof(float);
    
    cudaMalloc(&d_x, size);
    cudaMalloc(&d_y, size);
    cudaMalloc(&d_z, size);
    cudaMalloc(&d_masses, size);
    cudaMalloc(&d_potentials, size);
    
    // Copy data to device
    cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_z, h_z, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_masses, h_masses, size, cudaMemcpyHostToDevice);
    
    // Launch kernel
    int threads_per_block = 256;
    int blocks = (n_bodies + threads_per_block - 1) / threads_per_block;
    
    compute_potential_kernel<<<blocks, threads_per_block>>>(
        d_x, d_y, d_z, d_masses, d_potentials, n_bodies, G);
    
    // Wait for kernel to finish
    cudaDeviceSynchronize();
    
    // Copy results back to host
    cudaMemcpy(h_potentials, d_potentials, size, cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
    cudaFree(d_masses);
    cudaFree(d_potentials);
}

/*
 * Load data from binary file (positions only)
 * Format: plummer_data.bin (x1,y1,z1,x2,y2,z2,...)
 * Masses are computed as equal masses summing to 1.0 (m_i = 1/N)
 */
void load_data(const char *pos_file, float **x, float **y, float **z, float **masses, int *n_bodies) {
    FILE *fp;

    // Read positions
    fp = fopen(pos_file, "rb");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open positions file: %s\n", pos_file);
        exit(1);
    }

    // Read n_bodies from header (first 4 bytes as int32)
    int n_bodies_header;
    fread(&n_bodies_header, sizeof(int), 1, fp);
    *n_bodies = n_bodies_header;

    printf("Loading %d bodies from %s\n", *n_bodies, pos_file);

    // Allocate host memory
    *x = (float*)malloc(*n_bodies * sizeof(float));
    *y = (float*)malloc(*n_bodies * sizeof(float));
    *z = (float*)malloc(*n_bodies * sizeof(float));
    *masses = (float*)malloc(*n_bodies * sizeof(float));

    // Read separate arrays: [x:float32*n][y:float32*n][z:float32*n][masses:float32*n]
    fread(*x, sizeof(float), *n_bodies, fp);
    fread(*y, sizeof(float), *n_bodies, fp);
    fread(*z, sizeof(float), *n_bodies, fp);
    fread(*masses, sizeof(float), *n_bodies, fp);
    fclose(fp);

    printf("Loaded positions and masses from binary file\n");
}

/*
 * Main function
 */
int main(int argc, char **argv) {
    printf("CUDA Direct Summation\n");
    printf("=====================\n\n");

    // Default parameters
    const char *pos_file = "plummer_data.bin";
    const char *output_file = "outputs/direct_summation/potentials_cuda.bin";
    const char *json_output = NULL;
    float G = 1.0f;

    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--input") == 0 && i+1 < argc) {
            pos_file = argv[++i];
        } else if (strcmp(argv[i], "--output") == 0 && i+1 < argc) {
            output_file = argv[++i];
        } else if (strcmp(argv[i], "--json_output") == 0 && i+1 < argc) {
            json_output = argv[++i];
        } else if (strcmp(argv[i], "--G") == 0 && i+1 < argc) {
            G = atof(argv[++i]);
        }
    }

    // Load data
    float *x, *y, *z, *masses, *potentials;
    int n_bodies;

    load_data(pos_file, &x, &y, &z, &masses, &n_bodies);

    // Allocate output array
    potentials = (float*)malloc(n_bodies * sizeof(float));

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warmup runs
    printf("Warming up (3 runs)...\n");
    for (int i = 0; i < 3; i++) {
        compute_potential_direct(x, y, z, masses, potentials, n_bodies, G);
    }

    // Timed runs
    printf("Computing potentials using direct summation - 10 timed runs...\n");
    float times[10];
    for (int i = 0; i < 10; i++) {
        cudaEventRecord(start);
        compute_potential_direct(x, y, z, masses, potentials, n_bodies, G);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&times[i], start, stop);
        if ((i + 1) % 5 == 0) {
            printf("  Completed %d/10 runs\n", i + 1);
        }
    }

    // Calculate timing statistics
    float sum_time = 0.0f, sum_sq_time = 0.0f;
    float min_time = times[0], max_time = times[0];
    for (int i = 0; i < 10; i++) {
        sum_time += times[i];
        sum_sq_time += times[i] * times[i];
        if (times[i] < min_time) min_time = times[i];
        if (times[i] > max_time) max_time = times[i];
    }
    float mean_time = sum_time / 10.0f;
    float std_time = sqrtf(sum_sq_time / 10.0f - mean_time * mean_time);

    // Calculate potential statistics
    float sum_pot = 0.0f, sum_sq_pot = 0.0f;
    float min_pot = potentials[0], max_pot = potentials[0];

    for (int i = 0; i < n_bodies; i++) {
        sum_pot += potentials[i];
        sum_sq_pot += potentials[i] * potentials[i];
        if (potentials[i] < min_pot) min_pot = potentials[i];
        if (potentials[i] > max_pot) max_pot = potentials[i];
    }

    float mean_pot = sum_pot / n_bodies;
    float std_pot = sqrtf(sum_sq_pot / n_bodies - mean_pot * mean_pot);

    // Calculate total Plummer potential: U = 0.5 * sum(m_i * phi_i)
    float U_computed = 0.0f;
    for (int i = 0; i < n_bodies; i++) {
        U_computed += masses[i] * potentials[i];
    }
    U_computed *= 0.5f;

    float U_analytical = -0.294524f;
    float relative_error = fabsf(U_computed - U_analytical) / fabsf(U_analytical);

    // Save results
    FILE *fp = fopen(output_file, "wb");
    if (fp) {
        fwrite(potentials, sizeof(float), n_bodies, fp);
        fclose(fp);
        printf("\nResults saved to %s\n", output_file);
    } else {
        fprintf(stderr, "Warning: Could not save output file\n");
    }

    // Write JSON output if requested
    if (json_output) {
        write_json_direct(
            json_output,
            "cuda",
            n_bodies,
            G,
            SOFTENING,
            mean_time,
            std_time,
            min_time,
            max_time,
            10,  // n_trials
            U_computed,
            U_analytical,
            relative_error,
            mean_pot,
            std_pot,
            min_pot,
            max_pot,
            pos_file,
            output_file
        );
        printf("\nJSON results saved to %s\n", json_output);
    }

    // Print results
    printf("\n--- Particle Potential Statistics ---\n");
    printf("  Mean potential: %.6f\n", mean_pot);
    printf("  Std potential: %.6f\n", std_pot);
    printf("  Min potential: %.6f\n", min_pot);
    printf("  Max potential: %.6f\n", max_pot);
    printf("\n--- Total Plummer Potential ---\n");
    printf("  Computed U: %.6f\n", U_computed);
    printf("  Analytical U: %.6f\n", U_analytical);
    printf("  Relative error: %.6e (%.4f%%)\n", relative_error, relative_error * 100.0f);
    printf("\n--- Timing ---\n");
    printf("  Mean time: %.3f ± %.3f ms\n", mean_time, std_time);
    printf("  Min time: %.3f ms\n", min_time);
    printf("  Max time: %.3f ms\n", max_time);

    // Cleanup
    free(x);
    free(y);
    free(z);
    free(masses);
    free(potentials);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
