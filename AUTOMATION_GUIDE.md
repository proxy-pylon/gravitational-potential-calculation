# Automation System Guide

## Overview

This document describes the automated experiment system for running gravitational potential calculations across different methods, implementations, and parameters.

## Phase 1: JSON Output ✅ COMPLETE

All implementations now output structured JSON files with:
- Method and implementation type
- All parameters (N, G, softening, etc.)
- Timing statistics (mean, std, min, max)
- Results (U_computed, relative error)
- Potential statistics

### Files Modified:
1. `direct_summation/direct_cupy.py` - Added `--json_output` flag
2. `direct_summation/direct_cuda.cu` - Added `--json_output` flag
3. `monte_carlo/mc_cupy.py` - Added `--json_output` flag
4. `monte_carlo/mc_cuda.cu` - Added `--json_output` flag
5. `deep_learning/mlp_pytorch.py` - Added `--json_output` flag (prediction mode)
6. `utils/json_writer.h` - Created JSON writer helper for C/CUDA

## Phase 2: Configuration System

File: `experiment_config.py`

Defines all experimental configurations:
- N values to test: [10^4, 10^5, 10^6]
- Softening values: [0.001, 0.01, 0.1]
- MC sample counts: [10^3, 10^4, 10^5, 10^6, 10^7, 10^8]
- Methods: direct, monte_carlo, deep_learning
- Implementations: cupy, cuda (pytorch for DL)

## Phase 3: Experiment Runner

File: `run_experiments.py`

Automatically:
1. Generates data files for each N
2. Runs all method/implementation combinations
3. Collects JSON results
4. Aggregates into CSV summary
5. Handles errors gracefully

## Phase 4: Plotting System

File: `generate_plots.py`

Creates all required plots:
1. Runtime vs N (all methods)
2. Relative error vs N
3. MC: Error vs K (sample count)
4. Speedup comparison (CuPy vs CUDA)
5. DL: Training loss curves

## Usage

### 1. Run All Experiments

```bash
python run_experiments.py --output_dir experiments/run_001
```

### 2. Generate Plots

```bash
python generate_plots.py --results experiments/run_001/results.csv --output_dir experiments/run_001/plots
```

### 3. Run Specific Configurations

```bash
# Run only direct summation with CuPy
python run_experiments.py --methods direct --implementations cupy

# Run Monte Carlo sweep
python run_experiments.py --methods monte_carlo --sweep_mc_samples
```

## Output Structure

```
experiments/
└── run_YYYYMMDD_HHMMSS/
    ├── data/                   # Generated Plummer data
    │   ├── plummer_10000.npz
    │   ├── plummer_100000.npz
    │   └── plummer_1000000.npz
    ├── results/                # Individual JSON results
    │   ├── direct_cupy_N10000.json
    │   ├── direct_cuda_N10000.json
    │   ├── mc_cupy_N10000_K1000.json
    │   └── ...
    ├── results.csv             # Aggregated results
    ├── logs/                   # Execution logs
    │   └── experiment.log
    └── plots/                  # Generated plots
        ├── runtime_vs_N.png
        ├── error_vs_N.png
        ├── mc_error_vs_K.png
        └── speedup_comparison.png
```

## JSON Result Schema

### Direct Summation / Monte Carlo
```json
{
  "method": "direct_summation" | "monte_carlo",
  "implementation": "cupy" | "cuda",
  "timestamp": "ISO8601",
  "parameters": {
    "n_bodies": int,
    "G": float,
    "softening": float,
    "n_samples": int (MC only),
    "seed": int (MC only)
  },
  "timing": {
    "mean_time_seconds": float,
    "std_time_seconds": float,
    "min_time_seconds": float,
    "max_time_seconds": float,
    "n_trials": int,
    "warmup_runs": int
  },
  "results": {
    "U_computed": float,
    "U_analytical": float,
    "relative_error": float,
    "relative_error_percent": float
  },
  "potential_statistics": {
    "mean": float,
    "std": float,
    "min": float,
    "max": float
  },
  "files": {
    "input": str,
    "output": str
  }
}
```

### Deep Learning
```json
{
  "method": "deep_learning",
  "implementation": "pytorch",
  "timestamp": "ISO8601",
  "parameters": {
    "n_bodies": int,
    "model_file": str,
    "input_dim": int
  },
  "timing": {
    "inference_time_seconds": float,
    "n_predictions": int
  },
  "results": {
    "U_computed": float,
    "U_analytical": float,
    "relative_error": float,
    "relative_error_percent": float
  },
  "files": {
    "input": str,
    "model": str,
    "output": str
  }
}
```

## Next Steps

1. Create `experiment_config.py`
2. Create `run_experiments.py`
3. Create `generate_plots.py`
4. Test the full pipeline
5. Document results in final report
