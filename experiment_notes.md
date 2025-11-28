# Direct summation part

## With a hundred thousand bodies, system collapses without batch-sizing
```
root@6244a75d1df3:/workspace# python3 data_gen.py --n_bodies 100000   
Generating Plummer distribution with 100000 bodies...
  Scale radius: 1.0
  Random seed: 42
Saving npz data to plummer_data.npz...
Data saved to plummer_data.npz
  Positions shape: (100000, 3)
  Masses shape: (100000,)
  Position range: [-242.824, 393.730]
Saving binary data to plummer_data.bin...
Binary data saved to plummer_data.bin (100000 bodies)
Binary data saved to plummer_data.bin
root@6244a75d1df3:/workspace# python3 direct_summation/direct_cupy.py 
Loading data from plummer_data.npz...
  Number of bodies: 100000
Computing potentials using direct summation (CuPy)...
Traceback (most recent call last):
  File "/workspace/direct_summation/direct_cupy.py", line 138, in <module>
    main()
  File "/workspace/direct_summation/direct_cupy.py", line 124, in main
    potentials = compute_potential_direct(positions, masses, G=args.G)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/direct_summation/direct_cupy.py", line 84, in compute_potential_direct
    dr = positions_i - positions_j
         ~~~~~~~~~~~~^~~~~~~~~~~~~
  File "cupy/_core/core.pyx", line 1379, in cupy._core.core._ndarray_base.__sub__
  File "cupy/_core/_kernel.pyx", line 1349, in cupy._core._kernel.ufunc.__call__
  File "cupy/_core/_kernel.pyx", line 645, in cupy._core._kernel._get_out_args_from_optionals
  File "cupy/_core/core.pyx", line 2999, in cupy._core.core._ndarray_init
  File "cupy/_core/core.pyx", line 272, in cupy._core.core._ndarray_base._init_fast
  File "cupy/cuda/memory.pyx", line 875, in cupy.cuda.memory.alloc
  File "cupy/cuda/memory.pyx", line 1579, in cupy.cuda.memory.MemoryPool.malloc
  File "cupy/cuda/memory.pyx", line 1600, in cupy.cuda.memory.MemoryPool.malloc
  File "cupy/cuda/memory.pyx", line 1271, in cupy.cuda.memory.SingleDeviceMemoryPool.malloc
  File "cupy/cuda/memory.pyx", line 1292, in cupy.cuda.memory.SingleDeviceMemoryPool._malloc
  File "cupy/cuda/memory.pyx", line 1537, in cupy.cuda.memory.SingleDeviceMemoryPool._try_malloc
  File "cupy/cuda/memory.pyx", line 1540, in cupy.cuda.memory.SingleDeviceMemoryPool._try_malloc
cupy.cuda.memory.OutOfMemoryError: Out of memory allocating 120,000,000,000 bytes (allocated so far: 1,600,512 bytes).
root@6244a75d1df3:/workspace#                                                                                                              
```
120,000,000,000 bytes is about 112 gigabytes, so of course my system collapsed. 
```
root@6244a75d1df3:/workspace# free -h                                                                                                      
               total        used        free      shared  buff/cache   available
Mem:            31Gi       5.5Gi       7.1Gi        47Mi        18Gi        25Gi
Swap:             0B          0B          0B
```
I only have 25 gigabytes of RAM. So we need to implement batching

With batching, I could do 100,000! Yay!
When I started 1,000,000, it just got stuck :(. Takes too long
I increased chunk size from 1024 to 2**14

## Broadcasting slows things down
With 
```
dr = pos_i[:, None, :] - pos_j[None, :, :]
r = sqrt(sum(dr**2))
we get these results
```

```
root@6244a75d1df3:/workspace# python3 data_gen.py --n_bodies 100000
Generating Plummer distribution with 100000 bodies...
  Scale radius: 1.0
  Random seed: 42
Saving npz data to plummer_data.npz...
Data saved to plummer_data.npz
  Positions shape: (100000, 3)
  Masses shape: (100000,)
  Position range: [-242.824, 393.730]
Saving binary data to plummer_data.bin...
Binary data saved to plummer_data.bin (100000 bodies)
Binary data saved to plummer_data.bin
root@6244a75d1df3:/workspace# 
root@6244a75d1df3:/workspace# 
root@6244a75d1df3:/workspace# 
root@6244a75d1df3:/workspace# 
root@6244a75d1df3:/workspace# python3 direct_summation/direct_cupy.py 
Loading data from plummer_data.npz...
  Number of bodies: 100000
Computing potentials using direct summation (CuPy)...
Results saved to potentials_direct.npy
  Mean potential: -0.590002
  Std potential: 0.229817
  Min potential: -1.001224
  Max potential: -0.002303
Execution time: 9.049879 seconds
root@6244a75d1df3:/workspace# python3 direct_summation/direct_cupy.py 
Loading data from plummer_data.npz...
  Number of bodies: 100000
Computing potentials using direct summation (CuPy)...
Results saved to potentials_direct.npy
  Mean potential: -0.590002
  Std potential: 0.229817
  Min potential: -1.001224
  Max potential: -0.002303
Execution time: 9.072236 seconds
root@6244a75d1df3:/workspace# python3 direct_summation/direct_cupy.py 
Loading data from plummer_data.npz...
  Number of bodies: 100000
Computing potentials using direct summation (CuPy)...
Results saved to potentials_direct.npy
  Mean potential: -0.590002
  Std potential: 0.229817
  Min potential: -1.001224
  Max potential: -0.002303
Execution time: 9.059791 seconds
```

but with this, we get faster about 6 times
```
dx = pos_i[:, None, 0] - pos_j[None, :, 0]
dy = pos_i[:, None, 1] - pos_j[None, :, 1]
dz = pos_i[:, None, 2] - pos_j[None, :, 2]

r = cp.sqrt(dx*dx + dy*dy + dz*dz + softening_sq)
```

```
root@6244a75d1df3:/workspace# python3 direct_summation/direct_cupy.py 
Loading data from plummer_data.npz...
  Number of bodies: 100000
Computing potentials using direct summation (CuPy)...
Results saved to potentials_direct.npy
  Mean potential: -0.590002
  Std potential: 0.229817
  Min potential: -1.001224
  Max potential: -0.002303
Execution time: 1.525184 seconds
root@6244a75d1df3:/workspace# python3 direct_summation/direct_cupy.py 
Loading data from plummer_data.npz...
  Number of bodies: 100000
Computing potentials using direct summation (CuPy)...
Results saved to potentials_direct.npy
  Mean potential: -0.590002
  Std potential: 0.229817
  Min potential: -1.001224
  Max potential: -0.002303
Execution time: 1.497434 seconds
root@6244a75d1df3:/workspace# python3 direct_summation/direct_cupy.py 
Loading data from plummer_data.npz...
  Number of bodies: 100000
Computing potentials using direct summation (CuPy)...
Results saved to potentials_direct.npy
  Mean potential: -0.590002
  Std potential: 0.229817
  Min potential: -1.001224
  Max potential: -0.002303
Execution time: 1.499958 seconds
root@6244a75d1df3:/workspace# 
```

## Chunk size
With chunk size 2**16 I got OOM error. 
with chunk size 2**15 it was OK.
```
root@6244a75d1df3:/workspace# python3 direct_summation/direct_cupy.py 
Loading data from plummer_data.npz...
  Number of bodies: 100000
Computing potentials using direct summation (CuPy)...
Results saved to potentials_direct.npy
  Mean potential: -0.590002
  Std potential: 0.229817
  Min potential: -1.001224
  Max potential: -0.002303
Execution time: 2.041692 seconds
root@6244a75d1df3:/workspace# python3 direct_summation/direct_cupy.py 
Loading data from plummer_data.npz...
  Number of bodies: 100000
Computing potentials using direct summation (CuPy)...
Results saved to potentials_direct.npy
  Mean potential: -0.590002
  Std potential: 0.229817
  Min potential: -1.001224
  Max potential: -0.002303
Execution time: 1.477591 seconds
root@6244a75d1df3:/workspace# python3 direct_summation/direct_cupy.py 
Loading data from plummer_data.npz...
  Number of bodies: 100000
Computing potentials using direct summation (CuPy)...
Results saved to potentials_direct.npy
  Mean potential: -0.590002
  Std potential: 0.229817
  Min potential: -1.001224
  Max potential: -0.002303
Execution time: 1.481849 seconds
root@6244a75d1df3:/workspace# python3 direct_summation/direct_cupy.py 
Loading data from plummer_data.npz...
  Number of bodies: 100000
Computing potentials using direct summation (CuPy)...
Results saved to potentials_direct.npy
  Mean potential: -0.590002
  Std potential: 0.229817
  Min potential: -1.001224
  Max potential: -0.002303
Execution time: 1.477124 seconds
root@6244a75d1df3:/workspace# 
```

Now lets do chunk size 2**10. It worked slower because there were more I/O interactions and more overhead

```
root@6244a75d1df3:/workspace# python3 direct_summation/direct_cupy.py 
Loading data from plummer_data.npz...
  Number of bodies: 100000
Computing potentials using direct summation (CuPy)...
Results saved to potentials_direct.npy
  Mean potential: -0.590002
  Std potential: 0.229817
  Min potential: -1.001225
  Max potential: -0.002303
Execution time: 2.414499 seconds
root@6244a75d1df3:/workspace# python3 direct_summation/direct_cupy.py 
Loading data from plummer_data.npz...
  Number of bodies: 100000
Computing potentials using direct summation (CuPy)...
Results saved to potentials_direct.npy
  Mean potential: -0.590002
  Std potential: 0.229817
  Min potential: -1.001225
  Max potential: -0.002303
Execution time: 2.381820 seconds
root@6244a75d1df3:/workspace# python3 direct_summation/direct_cupy.py 
Loading data from plummer_data.npz...
  Number of bodies: 100000
Computing potentials using direct summation (CuPy)...
Results saved to potentials_direct.npy
  Mean potential: -0.590002
  Std potential: 0.229817
  Min potential: -1.001225
  Max potential: -0.002303
Execution time: 2.571073 seconds
```

## A million bodies
for a hundred thousand bodies it took 2 seconds. For a million it took 2 minutes

```
root@6244a75d1df3:/workspace# python3 direct_summation/direct_cupy.py 
Loading data from plummer_data.npz...
  Number of bodies: 1000000
Computing potentials using direct summation (CuPy)...
Results saved to potentials_direct.npy
  Mean potential: -0.588370
  Std potential: 0.229880
  Min potential: -0.998425
  Max potential: -0.001060
Execution time: 131.694573 seconds
```

I think the reason for such slow results is the double loops in the code. Switching to raw CUDA kernels would be much more effective. Let's see that in the direct_cuda.cu

## You don't need blocks in the cuda version
You don't do broadcasting and don't create a gigantic array. Each cuda thread handles one i and processes a linear array of j. So each thread touches all particles, but never builds an NxN array.

That part is incorrect where prof writes in the assignment that "Develop a tiled kernel in CUDA C where threads load position data into shared memory in blocks (e.g., 32x32 tile size) to reduce global memory accesses" 

# Monte Carlo part

## You can't use large number of samples K for python version because broadcasting will consume much of your RAM. 

But you can use large K for cuda version because it does not do broadcasting and does not consume much memory. Though of course it will be slower with larger K

## Error scaling

I read that relative error is inversely proportional to the square root of K. We need to check that experimentally. If it is so, then to reduce error 10 times you need 100 times more samples

I did with 1000 samples and got "Relative error: 1.153949e-03 (0.1154%)".
Then I did with 100000 samples and got "Relative error: 1.169633e-03 (0.1170%)".

There is almost no difference in error. In theory, if I increase number of samples 100 times, the error should get 10 times lower. But that did not happen. I suppose the reason is the systematic error because of epsilon. Indeed

Epsilon = 0.0001

```
root@5b56ba9c90d7:/workspace# ./build/mc_cuda --n_samples 1000
CUDA Monte Carlo Approximation
===============================

Loading 1000000 bodies from plummer_data.bin
Using equal masses: m_i = 1/1000000 = 1.000000e-06
Computing potentials using Monte Carlo (n_samples=1000)...

Results saved to outputs/monte_carlo/potentials_mc_cuda.bin

--- Particle Potential Statistics ---
  Mean potential: -0.588496
  Std potential: 0.230493
  Min potential: -1.556708
  Max potential: -0.001060

--- Total Plummer Potential ---
  Computed U: -0.294249
  Analytical U: -0.294524
  Relative error: 9.350791e-04 (0.0935%)

--- Timing ---
  Kernel execution time: 177.164 ms
```

Epsilon = 0.01
```
root@5b56ba9c90d7:/workspace# ./build/mc_cuda --n_samples 1000
CUDA Monte Carlo Approximation
===============================

Loading 1000000 bodies from plummer_data.bin
Using equal masses: m_i = 1/1000000 = 1.000000e-06
Computing potentials using Monte Carlo (n_samples=1000)...

Results saved to outputs/monte_carlo/potentials_mc_cuda.bin

--- Particle Potential Statistics ---
  Mean potential: -0.588363
  Std potential: 0.230348
  Min potential: -1.212197
  Max potential: -0.001060

--- Total Plummer Potential ---
  Computed U: -0.294184
  Analytical U: -0.294524
  Relative error: 1.153949e-03 (0.1154%)

--- Timing ---
  Kernel execution time: 51.734 ms
```

It means systematic error from epsilon choice eclipses statistical error from monte carlo

## Broadcasting leads to OOM

I tried more samples 

```
root@a0f25966bbaf:/workspace# python3 monte_carlo/mc_cupy.py --n_samples 10000
Loading data from plummer_data.npz...
  Number of bodies: 1000000
Warming up (3 runs)...
Traceback (most recent call last):
  File "/workspace/monte_carlo/mc_cupy.py", line 255, in <module>
    main()
  File "/workspace/monte_carlo/mc_cupy.py", line 152, in main
    _ = compute_potential_monte_carlo(
  File "/workspace/monte_carlo/mc_cupy.py", line 89, in compute_potential_monte_carlo
    sample_indices = cp.random.randint(0, n_bodies, size=(n_bodies, n_samples))
  File "/usr/local/lib/python3.10/dist-packages/cupy/random/_sample.py", line 110, in randint
    return rs.randint(low, high, size, dtype)
  File "/usr/local/lib/python3.10/dist-packages/cupy/random/_generator.py", line 1256, in randint
    x = self._interval(diff, size).astype(dtype, copy=False)
  File "/usr/local/lib/python3.10/dist-packages/cupy/random/_generator.py", line 714, in _interval
    sample = self._curand_generate(n_sample, dtype)
  File "/usr/local/lib/python3.10/dist-packages/cupy/random/_generator.py", line 762, in _curand_generate
    sample = cupy.empty((num,), dtype=dtype)
  File "/usr/local/lib/python3.10/dist-packages/cupy/_creation/basic.py", line 32, in empty
    return cupy.ndarray(shape, dtype, order=order)
  File "cupy/_core/core.pyx", line 167, in cupy._core.core.ndarray.__new__
  File "cupy/_core/core.pyx", line 254, in cupy._core.core._ndarray_base._init
  File "cupy/cuda/memory.pyx", line 875, in cupy.cuda.memory.alloc
  File "cupy/cuda/memory.pyx", line 1579, in cupy.cuda.memory.MemoryPool.malloc
  File "cupy/cuda/memory.pyx", line 1600, in cupy.cuda.memory.MemoryPool.malloc
  File "cupy/cuda/memory.pyx", line 1271, in cupy.cuda.memory.SingleDeviceMemoryPool.malloc
  File "cupy/cuda/memory.pyx", line 1292, in cupy.cuda.memory.SingleDeviceMemoryPool._malloc
  File "cupy/cuda/memory.pyx", line 1537, in cupy.cuda.memory.SingleDeviceMemoryPool._try_malloc
  File "cupy/cuda/memory.pyx", line 1540, in cupy.cuda.memory.SingleDeviceMemoryPool._try_malloc
cupy.cuda.memory.OutOfMemoryError: Out of memory allocating 40,000,000,000 bytes (allocated so far: 16,000,512 bytes).
root@a0f25966bbaf:/workspace# 
```

as you can see, 1000000 (n_bodies) * 10000 (n_samples) = 10e10
which is 37 GB if one element in float is 4 bytes.
Memory requirement scales as O(N*K)
so you need batches

## Problem with batches

```
python3 monte_carlo/mc_cupy.py --input plummer_data.npz --n_samples 10000000 --batch_size 10000
```

if 10e6 bodies and 10e4 samples (batch size)
shape = (1,000,000 bodies, 10,000 samples)
dtype = int64 (8 bytes)
total bytes = 1e6 * 1e4 * 8 = 80,000,000,000 bytes = 80 GB

So you can't sample for every body. It is incorrect that for every of 10e6 bodies gets its own 10000 samples 

# Deep learning part

The idea is that MLP will learn to "compress" the expensive pairwise computation into a fast NN inference. Then measure inference time

## Model does not generalize in N with raw particles

If you train on the N=5000, it will not generalize (will not even accept) N=10e6 because the model expects inputs of fixed size.

Let's see if it is good on N=5000 first, then see how it works with statistics

## Write interfaces first 

The professor first wrote interfaces of interaction. He specified what everything inputs and outputs. This is much better than what I do: doing everything step by step from scratch

## Aggregated statistics

Instead of loading the whole vector with N features, I preprocessed, feature engineered, and got only 21 most important features. That worked okay (though not within 1% error). Number of samples was 1k, number of bodies in a sample was 5k

```
root@5b56ba9c90d7:/workspace/deep_learning# python3 mlp_pytorch.py --train --input ../outputs/deep_learning/training_data.npz 
Using device: cuda

============================================================
TRAINING MODE
============================================================

Loading training data from ../outputs/deep_learning/training_data.npz...
  Loaded 1000 samples
  Positions shape: (1000, 5000, 3)
  U range: [-0.304107, -0.285098]

Splitting data: 800 train, 200 validation
Extracting features from 800 samples...
  Processed 100/800 samples
  Processed 200/800 samples
  Processed 300/800 samples
  Processed 400/800 samples
  Processed 500/800 samples
  Processed 600/800 samples
  Processed 700/800 samples
  Processed 800/800 samples
Feature extraction complete. Feature shape: torch.Size([800, 21])
Extracting features from 200 samples...
  Processed 100/200 samples
  Processed 200/200 samples
Feature extraction complete. Feature shape: torch.Size([200, 21])

Initializing MLP with input dimension: 21

Training on cuda...
Epochs: 100, Learning rate: 0.001
Early stopping threshold: 0.0001
------------------------------------------------------------
Epoch [1/100], Train Loss: 9.990511e-03, Val Loss: 1.766733e-03

Early stopping: validation loss 6.140788e-05 < 0.0001
------------------------------------------------------------
Training complete. Best validation loss: 1.027477e-04

Model saved to ../outputs/deep_learning/model.pt

============================================================
TRAINING RESULTS
============================================================
Train MSE: 9.124434e-05
Val MSE:   6.140788e-05
Train relative error (vs analytical): 2.5891%
Val relative error (vs analytical):   2.5173%
Analytical U: -0.294524
Predicted U mean (val): -0.291414
```

## Bigger sample size

Instead of trying on 1k samples, I try now to do 10k. 

```
root@5b56ba9c90d7:/workspace/deep_learning# python3 mlp_pytorch.py --train --input ../outputs/deep_learning/training_data.npz 
Using device: cuda

============================================================
TRAINING MODE
============================================================

Loading training data from ../outputs/deep_learning/training_data.npz...
  Loaded 10000 samples
  Positions shape: (10000, 5000, 3)
  U range: [-0.306281, -0.282778]

Splitting data: 8000 train, 2000 validation
Extracting features from 8000 samples...
  Processed 100/8000 samples
  Processed 200/8000 samples
  Processed 300/8000 samples
  Processed 400/8000 samples
  Processed 500/8000 samples
  Processed 600/8000 samples
  Processed 700/8000 samples
  Processed 800/8000 samples
  Processed 900/8000 samples
  Processed 1000/8000 samples
  Processed 1100/8000 samples
  Processed 1200/8000 samples
  Processed 1300/8000 samples
  Processed 1400/8000 samples
  Processed 1500/8000 samples
  Processed 1600/8000 samples
  Processed 1700/8000 samples
  Processed 1800/8000 samples
  Processed 1900/8000 samples
  Processed 2000/8000 samples
  Processed 2100/8000 samples
  Processed 2200/8000 samples
  Processed 2300/8000 samples
  Processed 2400/8000 samples
  Processed 2500/8000 samples
  Processed 2600/8000 samples
  Processed 2700/8000 samples
  Processed 2800/8000 samples
  Processed 2900/8000 samples
  Processed 3000/8000 samples
  Processed 3100/8000 samples
  Processed 3200/8000 samples
  Processed 3300/8000 samples
  Processed 3400/8000 samples
  Processed 3500/8000 samples
  Processed 3600/8000 samples
  Processed 3700/8000 samples
  Processed 3800/8000 samples
  Processed 3900/8000 samples
  Processed 4000/8000 samples
  Processed 4100/8000 samples
  Processed 4200/8000 samples
  Processed 4300/8000 samples
  Processed 4400/8000 samples
  Processed 4500/8000 samples
  Processed 4600/8000 samples
  Processed 4700/8000 samples
  Processed 4800/8000 samples
  Processed 4900/8000 samples
  Processed 5000/8000 samples
  Processed 5100/8000 samples
  Processed 5200/8000 samples
  Processed 5300/8000 samples
  Processed 5400/8000 samples
  Processed 5500/8000 samples
  Processed 5600/8000 samples
  Processed 5700/8000 samples
  Processed 5800/8000 samples
  Processed 5900/8000 samples
  Processed 6000/8000 samples
  Processed 6100/8000 samples
  Processed 6200/8000 samples
  Processed 6300/8000 samples
  Processed 6400/8000 samples
  Processed 6500/8000 samples
  Processed 6600/8000 samples
  Processed 6700/8000 samples
  Processed 6800/8000 samples
  Processed 6900/8000 samples
  Processed 7000/8000 samples
  Processed 7100/8000 samples
  Processed 7200/8000 samples
  Processed 7300/8000 samples
  Processed 7400/8000 samples
  Processed 7500/8000 samples
  Processed 7600/8000 samples
  Processed 7700/8000 samples
  Processed 7800/8000 samples
  Processed 7900/8000 samples
  Processed 8000/8000 samples
Feature extraction complete. Feature shape: torch.Size([8000, 21])
Extracting features from 2000 samples...
  Processed 100/2000 samples
  Processed 200/2000 samples
  Processed 300/2000 samples
  Processed 400/2000 samples
  Processed 500/2000 samples
  Processed 600/2000 samples
  Processed 700/2000 samples
  Processed 800/2000 samples
  Processed 900/2000 samples
  Processed 1000/2000 samples
  Processed 1100/2000 samples
  Processed 1200/2000 samples
  Processed 1300/2000 samples
  Processed 1400/2000 samples
  Processed 1500/2000 samples
  Processed 1600/2000 samples
  Processed 1700/2000 samples
  Processed 1800/2000 samples
  Processed 1900/2000 samples
  Processed 2000/2000 samples
Feature extraction complete. Feature shape: torch.Size([2000, 21])

Initializing MLP with input dimension: 21

Training on cuda...
Epochs: 100, Learning rate: 0.001
Early stopping threshold: 0.0001
------------------------------------------------------------
Epoch [1/100], Train Loss: 7.176627e-03, Val Loss: 1.435341e-04

Early stopping: validation loss 5.552300e-05 < 0.0001
------------------------------------------------------------
Training complete. Best validation loss: 1.250567e-04

Model saved to ../outputs/deep_learning/model.pt

============================================================
TRAINING RESULTS
============================================================
Train MSE: 1.405499e-04
Val MSE:   5.552299e-05
Train relative error (vs analytical): 1.7995%
Val relative error (vs analytical):   1.8159%
Analytical U: -0.294524
Predicted U mean (val): -0.290236
```

As can be noticed, the error indeed got lower just because the model had 10 times more data. The quality of data though is not good. The algorithm use epsilon 0.001 which resulted in poor U estimates. The MLP fitted to those poor estimates also predicts U poorly, no surprise. To decrease the error further, use higher-quality data and better U estimates.


