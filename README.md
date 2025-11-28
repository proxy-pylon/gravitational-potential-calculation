[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/NPTsFQCk)
# PHYS 421: Parallel Computing (Fall 2025)
## Assignment #7: Gravitational Potential Calculation for Star Clusters

**Invite link:** https://classroom.github.com/a/CPNCLXDU

**Due date:** Monday, November 24 at 23:59

**Deadline extension:** All assignments are automatically granted a 24-hour extension to resolve any possible technical or personal problems

**Points:** 100 points

---

## Purpose

- Implement gravitational potential energy U via direct summation, Monte Carlo (MC), and deep learning (DL) approximation.
- Benchmark CUDA kernels with events (end-to-end and kernel-only times).
- Compare methods (Direct vs. MC vs. DL) and implementations (CUDA C vs. CuPy).
- Conduct scaling study on synthetic Plummer data for N=1e4, 1e5, 1e6.
- Validate with tolerances, sample counts, and training metrics.

---

## Introduction and Background

In astrophysics and computational physics, the gravitational potential energy $U$ of a system of particles is a fundamental quantity that quantifies the work required to assemble the system from infinity, given by the formula

$$U = -\sum_{i < j} \frac{G m_i m_j}{r_{ij}},$$

where $G$ is the gravitational constant, $m_i$ and $m_j$ are the masses of particles $i$ and $j$, and $r_{ij} = |\mathbf{r}_i - \mathbf{r}_j|$ is the distance between their positions. This pairwise interaction underpins the dynamics of self-gravitating systems like star clusters, where $U$ balances kinetic energy according to the virial theorem, influencing stability, relaxation, and evolution over time.

A canonical model for such systems is the **Plummer sphere**, introduced by H.C. Plummer in 1911 as an analytical approximation for globular clusters. It describes a spherically symmetric density distribution

$$\rho(r) = \frac{3M}{4\pi a^3} \left(1 + \frac{r^2}{a^2}\right)^{-5/2},$$

where $M$ is the total mass and $a$ is the scale radius defining the core size. The model features a dense central region transitioning to an extended halo, is finite in mass despite infinite extent, and avoids a central cusp, making it numerically stable. Crucially, the continuous Plummer model admits an exact analytical solution for the potential energy: for normalized parameters $G = 1$, $M = 1$, and $a = 1$,

$$U = -\frac{3\pi}{32} \approx -0.294524.$$

This closed-form result serves as a benchmark for validating discrete approximations.

In simulations, the continuous density is sampled with $N$ particles of equal mass $m = 1/N$, turning $U$ into a discrete sum. Direct summation evaluates all $N(N-1)/2$ pairs exactly but scales as $O(N^2)$, becoming infeasible for large $N$ (e.g., $10^6$) without acceleration. Monte Carlo methods approximate $U$ by sampling $K \ll N^2$ random pairs, yielding an unbiased estimator with statistical error scaling as $1/\sqrt{K}$, though heavy-tailed distributions from close encounters require softening $\epsilon$ (replacing $1/r$ with $1/\sqrt{r^2 + \epsilon^2}$) to control variance.

Deep learning offers a modern alternative: neural networks, such as multilayer perceptrons (MLPs), learn mappings from particle positions to $U$, trained on labels from small-scale direct computations. Inference is efficient (e.g., $O(N)$ for aggregated inputs), capturing nonlinear patterns while amortizing costs. An optional extension is physics-informed neural networks (PINNs), which embed gravitational equations into the loss function for improved physical consistency and generalization.

This assignment explores these methods in the context of GPU parallelism, leveraging CUDA for high-performance computing in scientific simulations.

---

## Overview & Policy

- **Methods to Implement:** You must implement three methods for calculating U: Direct Summation (exact pairwise computation), Monte Carlo or MC (stochastic sampling with K=1e8 as a baseline), and Deep Learning or DL (using a simple multilayer perceptron or MLP model).
- **Implementations Required:** For each method, provide two GPU-based versions: one using custom CUDA C kernels (low-level for fine-grained control and optimization) and one using CuPy (high-level Pythonic interface for vectorized operations). For the DL method, you may use either CuPy to implement custom neurons and layers from scratch (e.g., matrix multiplications and activations on GPU) or PyTorch (preferred for its built-in support for neural networks, with CUDA enabled for GPU acceleration).
- **Comparisons to Perform:** Analyze and compare the accuracy (e.g., error relative to analytical value) and speed (e.g., runtime in seconds) of the three methods; also compare the two implementations (CUDA C vs. CuPy) in terms of performance, code complexity, and ease of debugging.
- **Bonus Opportunities (+0–5 pts):** Earn extra points by implementing advanced DL variants, such as physics-informed neural networks (PINNs) or graph neural networks (GNNs), or by adding optimizations like improved load balancing in kernels or variance reduction techniques in MC. Clearly label any bonus work as "optional" in your code and report.
- **Starter Repository:** Use the GitHub Classroom invite link provided above to accept the assignment and get your personal repository. The template repository includes starter code skeletons for CUDA C and CuPy implementations, a basic data generation script for Plummer sampling, and a simple PyTorch MLP example. Fork or clone your repository as needed for version control; make regular commits to track your progress and include commit history in your submission for reproducibility.

---

```
assignment-7-starter/
├── README.md                  # This file (instructions for students)
├── data_gen.py                # Python script for Plummer data generation
├── direct_summation/          # Folder for direct summation skeletons
│   ├── direct_cuda.cu         # CUDA C kernel skeleton
│   └── direct_cupy.py         # CuPy script skeleton
├── monte_carlo/               # Folder for MC skeletons
│   ├── mc_cuda.cu             # CUDA C kernel skeleton
│   └── mc_cupy.py             # CuPy script skeleton
├── deep_learning/             # Folder for DL skeletons
│   ├── mlp_pytorch.py         # PyTorch MLP skeleton
│   └── mlp_cupy.py            # CuPy from-scratch MLP skeleton (optional/advanced)
├── utils/                     # Optional utils
│   └── timing_utils.h         # CUDA timing helper (for events)
└── requirements.txt           # Python dependencies
```

## Getting Started

## Project Structure

## Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (required for all parts of this assignment)
- CUDA Toolkit 11.x or 12.x
- Basic knowledge of CUDA programming and Python

### Installation

1. Clone this repository
2. Install Python dependencies:

```bash
pip install numpy

# For GPU acceleration with CuPy (choose based on your CUDA version):
pip install cupy-cuda11x  # For CUDA 11.x
# OR
pip install cupy-cuda12x  # For CUDA 12.x

# For deep learning:
pip install torch torchvision
```

Alternatively, use the requirements file (uncomment the appropriate CuPy version):

```bash
pip install -r requirements.txt
```

---

## Task: Potential Calculation

Your main task is to implement the three methods and compare them for particle counts **N=10,000 (1e4), 100,000 (1e5), and 1,000,000 (1e6)** using synthetic data generated from the Plummer model. Follow the detailed instructions below for each part, ensuring all code runs on GPU and is optimized for performance.

### Common Details for All Parts

**Softening Parameter:**
- Always incorporate gravitational softening with **ε=0.01** to avoid numerical singularities in close particle encounters
- This replaces $1/r_{ij}$ with $1/\sqrt{r_{ij}^2 + \epsilon^2}$ in all calculations
- Test different ε values (e.g., 0.001 to 0.1) in your analysis to see how it affects accuracy and variance

**Validation Process:**
- For every run, compute the relative error as $|U_{computed} - U_{analytical}| / |U_{analytical}|$, where **$U_{analytical} = -0.294524$**
- Aim for tolerances of **1e-5 or better** for Direct and MC methods (indicating high precision)
- Aim for **1% or better** for DL (allowing for approximation error)
- Report these errors in a table for each N and method, and discuss any deviations

**Scaling Study Requirements:**
- Execute each method and implementation for all three N values
- Measure and plot how runtime and error scale with N
- Include discussions on computational complexity, memory usage, and practical limits

**Optimizations and Tips:**
- Prioritize coalesced memory access by structuring position data as Structure of Arrays (SoA, e.g., separate arrays for x, y, z coordinates) rather than Array of Structures
- Minimize host-to-device (H2D) and device-to-host (D2H) data transfers by keeping large arrays on the GPU throughout computations
- Use **float32** data type for faster performance unless precision issues arise (then justify switching to float64)
- Profile your kernels using tools like nvprof or nsys to analyze warp occupancy, memory bandwidth, and bottlenecks
- For large N, use chunking or streaming to manage VRAM limits
- Always test your code on small N (e.g., 100 particles) first to debug logic before scaling up

### Timing Instructions (for All Parts)

**End-to-End Time:**
- Measure the total time, including data transfer from host to device (H2D), the core computation, and transfer back from device to host (D2H)
- For the DL method, report training time separately from inference, as training is a one-time cost

**Kernel-Only Time:**
- Focus solely on the GPU computation time, excluding data transfers
- Use CUDA events to record timestamps immediately before and after kernel launches

**Measurement Protocol:**
- Perform 3 warm-up runs (discard these to account for initial overheads like JIT compilation)
- Then run 10 timed trials and report the mean time ± standard deviation
- Ensure a stable environment by running on the same GPU, avoiding other processes, and fixing GPU clock speeds if possible via nvidia-smi

### Data Generation Instructions (for All Parts)

**Random Seed and Parameters:**
- Use a fixed seed (e.g., **42**, report it) for reproducibility in all random operations (e.g., via np.random.seed(42) or equivalent in CuPy/PyTorch)
- Work in 3D space for positions
- Set **G=1**, total mass **M=1**, scale radius **a=1**
- Assign equal masses **m=1/N** to each particle

**Sampling Process:**
- Generate radii $r$ using the inverse cumulative distribution function (CDF): $r = a (u^{-2/3} - 1)^{-1/2}$, where $u$ is uniform random in [0,1]
- Generate angles $\theta = \arccos{(2v - 1)}$ and $\phi = 2\pi w$, with $v$ and $w$ uniform in [0,1]
- Convert to Cartesian coordinates: $x = r \sin\theta \cos\phi$, $y = r \sin\theta \sin\phi$, $z = r \cos\theta$
- Truncate any $r > 100a$ by resampling (expect ~1% rejection rate)

**Subsampling and Storage:**
- First generate a large dataset at N=1e6
- Then subsample without replacement for smaller N values to ensure consistency across scales
- Store positions as float32 arrays to save memory

**Tips:**
- Use NumPy or CuPy for efficient generation
- Verify your sampling by checking statistics like mean particle radius (~0.75a) and total mass sum (=1)
- For DL, create subsamples (e.g., 1000 batches of N=1e3 to 1e4 particles each) for training/validation
- Consider data augmentation like random rotations to improve model robustness

**Reporting Requirements:**
- In your report, include the seed used, the N values tested, the softening ε, the analytical U value for reference
- For DL, include hyperparameters like number of layers, epochs trained, and final loss

---

## Assignment Tasks

### Task 0: Generate Data

First, generate N-body data using the Plummer distribution:

```bash
python data_gen.py --n_bodies 10000 --scale_radius 1.0 --seed 42 --output plummer_data.npz
```

This creates a file `plummer_data.npz` containing particle positions and masses.

---

### Part 1 (40 pts): Direct Summation

Implement the direct summation method in both CUDA C and CuPy.

**Theory:** The gravitational potential energy is:
$$U = -\sum_{i < j} \frac{G m_i m_j}{\sqrt{r_{ij}^2 + \epsilon^2}}$$

#### 1a. CUDA C Implementation

File: `direct_summation/direct_cuda.cu`

**Implementation Details:**
- Develop a tiled kernel in CUDA C where threads load position data into shared memory in blocks (e.g., 32x32 tile size) to reduce global memory accesses
- Compute pairwise distances and sum contributions using reductions (e.g., `__syncthreads()` for synchronization)
- Exclude self-interactions with conditional checks like `if (tid != jid)`

**TODO:**
- Complete the `compute_potential_kernel` function
- Each thread computes the potential for one particle
- Use softening parameter ε=0.01 to avoid singularities

**Compilation:**
```bash
cd direct_summation
nvcc -o direct_cuda direct_cuda.cu
```

**Tips:**
- Use `blockIdx.x * blockDim.x + threadIdx.x` to get particle index
- Avoid storing the full $N^2$ distance matrix, as it will cause memory overflow for $N=1e6$ (use on-the-fly computation instead)
- If your reduction uses `atomicAdd`, be aware of potential race conditions and test for correctness
- Benchmark different tile sizes (e.g., 16x16 vs. 64x64) to find the optimal for your GPU
- If your sum includes diagonal terms, correct by adding $+1/(2\epsilon)$ to account for self-potential

#### 1b. CuPy Implementation

File: `direct_summation/direct_cupy.py`

**Implementation Details:**
- Use broadcasting for efficient vectorization, but chunk the computation (e.g., process 128-1024 particles at a time) to avoid out-of-memory errors
- Example code approach: 
  ```python
  dx = pos[:, None, :] - pos_i[None, :, :]
  r2 = cp.sum(dx**2, axis=2) + eps2
  contrib = -m[:, None] / cp.sqrt(r2)
  ```
- Then sum `contrib` and scale by the appropriate factor

**TODO:**
- Complete the `compute_potential_direct` function
- Use CuPy array operations and broadcasting
- Leverage GPU parallelism through vectorization

**Run:**
```bash
python direct_summation/direct_cupy.py --input plummer_data.npz --output potentials_direct.npy
```

**Tips:**
- Use `cp.newaxis` for broadcasting
- Calculate all pairwise distances efficiently using broadcasting
- Use `cp.fill_diagonal()` to avoid self-interactions
- Chunk the computation for large N to manage memory

---

### Part 2 (40 pts): Monte Carlo Approximation

Implement Monte Carlo approximation in both CUDA C and CuPy.

**Theory:** Instead of summing all particles, randomly sample a subset:
$$U \approx -\frac{N(N-1)}{2K} \sum_{k=1}^{K} \frac{G m_i m_j}{\sqrt{r_{ij}^2 + \epsilon^2}}$$

where K is the number of random pair samples.

#### 2a. CUDA C Implementation

File: `monte_carlo/mc_cuda.cu`

**Implementation Details:**
- Generate random pair indices on the GPU using `cuRAND` in CUDA C
- Compute contributions in parallel for each sampled pair
- Aggregate the sum using atomic operations or parallel reductions

**TODO:**
- Complete the `compute_potential_mc_kernel` function
- Use cuRAND for random sampling
- Avoid sampling the same particle (self-interaction)
- Start with K=1e8 samples as baseline

**Compilation:**
```bash
cd monte_carlo
nvcc -o mc_cuda mc_cuda.cu
```

**Tips:**
- Initialize cuRAND state with `curand_init()`
- Use `curand()` to generate random indices
- Scale result by (N-1)/n_samples factor appropriately
- Initialize `cuRAND` states per thread with `curand_init` to ensure independent randomness
- Consider optional variance reduction techniques like stratified sampling or oversampling near-core particles

**Variance Estimation:**
- Mask out cases where `i==j` to avoid self-pairs
- Run the approximation with 5-10 different random seeds to estimate variance
- Report the standard deviation of $U$ estimates

**Parameter Tuning:**
- Start with $K=1e7$ samples
- Adjust $K$ upward if accuracy is below 1% error (e.g., try $K=1e9$ for better precision)
- Discuss in your report how error behaves as $\sim 1/\sqrt{K}$ and how heavy tails from close pairs affect results

#### 2b. CuPy Implementation

File: `monte_carlo/mc_cupy.py`

**TODO:**
- Complete the `compute_potential_monte_carlo` function
- Use `cp.random.randint()` for sampling
- Apply appropriate scaling

**Run:**
```bash
python monte_carlo/mc_cupy.py --input plummer_data.npz --output potentials_mc.npy --n_samples 100000000
```

---

### Part 3 (20 pts): Deep Learning (MLP)

Implement a Multi-Layer Perceptron to predict gravitational potentials.

File: `deep_learning/mlp_pytorch.py` (recommended) or `deep_learning/mlp_cupy.py` (advanced)

**Implementation Details:**
- Build a multilayer perceptron (MLP) with 3-5 hidden layers, 128-256 units per layer, and ReLU activations
- Input is the particle positions (N x 3, either flattened into a vector or aggregated with statistics like mean/std for scalar output $U$)
- Output is a single scalar $U$
- You may implement this using either:
  - **CuPy** (building custom neurons and layers from scratch, such as GPU-accelerated matrix multiplications for weights and biases, and activation functions)
  - **PyTorch** (recommended for its higher-level API, with CUDA support via `model.to('cuda')`)

**Training Process:**
- Train the model supervised on labels generated from direct summation on small subsamples (N=1e3 to 1e4 particles, creating 80% train / 20% validation split)
- Use the Adam optimizer with learning rate lr=1e-3
- Minimize mean squared error (MSE) loss
- Train for up to 100 epochs, but implement early stopping if validation loss drops below 1e-4 to avoid overfitting

**TODO:**
- Complete the `GravitationalPotentialMLP` class
- Implement the training loop
- Implement the prediction function

**Training:**
```bash
# First, generate ground truth using direct summation
python direct_summation/direct_cupy.py --input plummer_data.npz --output potentials_direct.npy

# Then train the MLP
python deep_learning/mlp_pytorch.py --train --input plummer_data.npz --ground_truth potentials_direct.npy --epochs 100 --model model.pt
```

**Prediction:**
```bash
python deep_learning/mlp_pytorch.py --predict --input test_data.npz --model model.pt --output predictions.npy
```

**Tips:**
- Use 4 input features: x, y, z, mass
- Try architecture: [128, 256, 128, 64] hidden layers
- Use ReLU activation and dropout (rate 0.1) for regularization
- MSE loss works well for this problem
- Normalize inputs by subtracting the mean and dividing by the standard deviation for positions to improve training stability
- Use batch sizes of 32-128 for efficient GPU utilization
- Monitor training with validation sets to detect overfitting
- For inference on large N, use batching to process in chunks
- In PyTorch, leverage `DataLoader` for data handling, `nn.MSELoss` for the objective, and `optim.Adam` for updates

**Optional PINN Extension (+bonus):**
- For a bonus, incorporate physics-informed terms in the loss function
- Add penalties for violating gravitational potential gradients $(\nabla U)$ or energy conservation laws

**Evaluation:**
- Compute test MSE on a held-out set of particles
- Discuss issues like overfitting (mitigate with dropout) and model transferability to larger N than trained on

---

## Submission Requirements

**Plots to Include:**
- Runtime vs. N for each method and implementation (line plots with error bars for std)
- Speedup factors relative to a baseline (e.g., direct in CuPy)
- Error vs. K for MC or vs. epochs for DL (showing convergence)

**Analysis Requirements:**
- Provide a detailed discussion of tradeoffs (e.g., accuracy vs. speed) and bottlenecks (e.g., memory limits in direct summation)
- Include code snippets or pseudocode where helpful

**Repository Submission:**
- Your GitHub repo should have a README.md with step-by-step run instructions (e.g., "nvcc compile command" or "python script.py")
- Organize code into folders (e.g., /direct_summation, /monte_carlo, /deep_learning)
- Include CSV files for raw timing/error data
- Provide plotting scripts (e.g., in Matplotlib) to reproduce figures

**Report PDF:**
- Name it `report_a7_<lastname>.pdf`
- Include environment details like GPU model, driver version, CUDA toolkit
- List all seeds, N values, K for MC, ε, and DL hyperparameters
- Embed all required plots and tables

---

## Grading (100 pts + bonus)

- **Part 1: 40 pts** (evaluated on correctness of implementation, optimization quality, validation results, and clarity of code/comments)
- **Part 2: 40 pts** (evaluated on handling of randomness, variance estimation, tuning of K, and discussion of convergence)
- **Part 3: 20 pts** (evaluated on model design, training process, evaluation metrics, and optional extensions like PINN)
- **Bonus: +0–5 pts** for advanced features (e.g., PINN implementation or novel optimizations, with clear documentation)

---

## Checklist

- [ ] All implementations (CUDA C and CuPy for each part) are complete and run without errors
- [ ] Scaling study conducted for all N values (1e4, 1e5, 1e6); timing and error data collected and validated
- [ ] Report includes all required elements: environment details, commands to reproduce, figures, and analysis
- [ ] Repository is well-organized with clear README instructions

---

## Notes & Tips

- Focus your efforts on GPU-specific optimizations: ensure coalesced reads by proper data alignment; use shared memory for reductions to speed up summations; keep large datasets on the device to avoid unnecessary transfers
- Tune parameters like K in MC or epochs in DL based on preliminary tests to balance accuracy and runtime
- For DL: Start with a simple MLP architecture to get baseline results; use PyTorch for quicker prototyping unless you want the challenge of CuPy from-scratch; subsample training data aggressively to keep training times reasonable
- Use float32 for all computations to maximize speed, but check for precision issues in validation
- Profile your kernels early to identify and fix bottlenecks

---

## References

- Plummer, H. C. (1911). "On the problem of distribution in globular star clusters"
- Barnes, J., & Hut, P. (1986). "A hierarchical O(N log N) force-calculation algorithm"
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CuPy Documentation](https://docs.cupy.dev/en/stable/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Hybrid integration of the gravitational N-body problem with Artificial Neural Networks](https://ml4physicalsciences.github.io/2022/files/NeurIPS_ML4PS_2022_88.pdf)
- [Efficient n-body simulations using physics-informed graph neural networks](https://arxiv.org/pdf/2504.01169)
- [Simple lessons from complex learning: what a neural network model learns about cosmic structure formation](https://doi.org/10.1093/pnasnexus/pgac250)
---

## Getting Help

- Review the hints and tips in each skeleton file
- Check CUDA error messages with proper error handling
- Compare your results with the analytical Plummer potential: **U = -0.294524**
- Start with small N (e.g., 100) for debugging
- Use profiling tools (nvprof, nsys) to identify performance bottlenecks

Good luck!
