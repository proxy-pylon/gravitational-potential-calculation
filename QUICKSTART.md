# Quick Start Guide

## Complete Automation System Implementation

All files have been successfully updated with JSON output capabilities and a complete automation system has been created.

## ✅ What's Been Done

### Phase 1: JSON Output (COMPLETE)
All implementations now output structured JSON:
- ✅ `direct_summation/direct_cupy.py` - Added `--json_output` flag + timing
- ✅ `direct_summation/direct_cuda.cu` - Added `--json_output` flag + timing
- ✅ `monte_carlo/mc_cupy.py` - Added `--json_output` flag + timing
- ✅ `monte_carlo/mc_cuda.cu` - Added `--json_output` flag + timing
- ✅ `deep_learning/mlp_pytorch.py` - Added `--json_output` flag + timing
- ✅ `utils/json_writer.h` - C/CUDA JSON writer utility

All implementations now perform:
- 3 warmup runs
- 10 timed runs for statistics
- Output mean ± std timing
- Structured JSON with all parameters and results

### Phase 2: Configuration System (COMPLETE)
- ✅ `experiment_config.py` - Centralized configuration for all experiments

### Phase 3: Automation Runner (COMPLETE)
- ✅ `run_experiments.py` - Automated experiment orchestration

### Phase 4: Plotting System (COMPLETE)
- ✅ `generate_plots.py` - Automatic plot generation from results

### Documentation (COMPLETE)
- ✅ `AUTOMATION_GUIDE.md` - Complete system documentation
- ✅ `QUICKSTART.md` - This file

## 🚀 Quick Start

### Step 1: View Configuration

```bash
python experiment_config.py
```

This shows you all parameters that will be tested. You can edit `experiment_config.py` to customize:
- N values: `[10000, 100000, 1000000]`
- Softening: `[0.001, 0.01, 0.1]`
- MC samples: `[1e3, 1e4, 1e5, 1e6, 1e7, 1e8]`
- Which methods to run

### Step 2: Test a Single Experiment

Before running everything, test one experiment:

```bash
# Test direct summation with CuPy
python direct_summation/direct_cupy.py \
    --input plummer_data.npz \
    --json_output test_result.json \
    --softening 0.01

# View the JSON output
cat test_result.json
```

### Step 3: Dry Run (see what would execute)

```bash
python run_experiments.py --dry_run
```

This shows you all experiments that would run without actually executing them.

### Step 4: Run Selected Experiments

```bash
# Run only direct summation
python run_experiments.py --methods direct

# Run direct summation and Monte Carlo
python run_experiments.py --methods direct monte_carlo

# Run everything with custom output directory
python run_experiments.py --output_dir experiments/my_experiment
```

### Step 5: Generate Plots

After experiments complete:

```bash
python generate_plots.py \
    --results experiments/run_YYYYMMDD_HHMMSS/results.csv \
    --output_dir experiments/run_YYYYMMDD_HHMMSS/plots
```

## 📊 Output Structure

After running experiments, you'll have:

```
experiments/run_YYYYMMDD_HHMMSS/
├── data/                          # Generated Plummer data
│   ├── plummer_10000.npz
│   ├── plummer_10000.bin         # For CUDA
│   ├── plummer_100000.npz
│   ├── plummer_100000.bin
│   ├── plummer_1000000.npz
│   └── plummer_1000000.bin
├── results/                       # Individual JSON results
│   ├── directcupy_N10000_eps1e-2.json
│   ├── directcuda_N10000_eps1e-2.json
│   ├── mccupy_N10000_K1000_eps1e-2.json
│   └── ... (many more)
├── results.csv                    # Aggregated CSV
├── logs/
│   └── experiment.log            # Execution log
└── plots/                         # Generated plots
    ├── runtime_vs_N.png
    ├── runtime_vs_N.pdf
    ├── error_vs_N.png
    ├── mc_error_vs_K.png
    ├── speedup_cupy_vs_cuda.png
    └── summary_table.png
```

## 🔧 Customization

### Modify Experiment Parameters

Edit `experiment_config.py`:

```python
# Test fewer N values (faster)
N_VALUES = [10000, 100000]

# Test fewer MC sample counts
MC_SAMPLE_COUNTS = [1000, 100000, 10000000]

# Disable specific methods
METHODS = {
    "direct_summation": True,
    "monte_carlo": False,  # Skip MC
    "deep_learning": True,
}
```

### Run Specific Parameter Sweeps

The system automatically tests all combinations of:
- N values × Softening values for direct summation
- N values × MC samples × Softening values for Monte Carlo
- N values for deep learning

You can filter by method using `--methods` flag.

## 📈 Understanding the Results

### JSON Output Structure

Each experiment produces a JSON file with:

```json
{
  "method": "direct_summation",
  "implementation": "cupy",
  "timestamp": "2025-11-26T12:00:00",
  "parameters": {
    "n_bodies": 100000,
    "G": 1.0,
    "softening": 0.01
  },
  "timing": {
    "mean_time_seconds": 1.523,
    "std_time_seconds": 0.045,
    "min_time_seconds": 1.477,
    "max_time_seconds": 1.591,
    "n_trials": 10,
    "warmup_runs": 3
  },
  "results": {
    "U_computed": -0.294123,
    "U_analytical": -0.294524,
    "relative_error": 0.001361,
    "relative_error_percent": 0.1361
  }
}
```

### CSV Aggregation

All JSON files are automatically aggregated into `results.csv`:

| method | implementation | n_bodies | softening | mean_time_s | relative_error | ... |
|--------|---------------|----------|-----------|-------------|----------------|-----|
| direct_summation | cupy | 10000 | 0.01 | 0.145 | 0.000012 | ... |
| direct_summation | cuda | 10000 | 0.01 | 0.052 | 0.000012 | ... |
| monte_carlo | cupy | 10000 | 0.01 | 0.025 | 0.0012 | ... |

## 🎯 Expected Results

Based on your experiment notes:

### Direct Summation
- **N=100k**: ~1.5 seconds (CuPy), faster with CUDA
- **N=1M**: ~2 minutes (CuPy)
- **Error**: < 1e-5 (very accurate)

### Monte Carlo
- **Runtime**: Much faster than direct for large N
- **Error**: Depends on K (samples)
  - K=1e3: ~1% error
  - K=1e8: < 0.1% error
- **Scaling**: Error ~ 1/√K

### Deep Learning
- **Training**: One-time cost on 10k samples
- **Inference**: Very fast (< 1 second)
- **Error**: ~1-2% (acceptable for fast approximation)

## ⚠️ Important Notes

### CUDA Compilation
The runner will automatically compile CUDA programs if needed. Ensure you have `nvcc` installed.

### Memory Limitations
- Direct summation: May OOM for N=1M if chunk_size too large
- Monte Carlo (CuPy): Limited by K×N memory for random sampling
- Adjust `chunk_size` in config if needed

### Deep Learning Model
The runner expects a trained model at `outputs/deep_learning/model.pt`.
If you haven't trained one yet, train it first:

```bash
# Generate training data (if not already done)
python deep_learning/generate_training_data.py \
    --n_samples 10000 \
    --n_bodies 5000

# Train model
python deep_learning/mlp_pytorch.py \
    --train \
    --input outputs/deep_learning/training_data.npz \
    --epochs 100
```

## 🐛 Troubleshooting

### "CuPy not installed"
```bash
pip install cupy-cuda11x  # or cupy-cuda12x
```

### "nvcc: command not found"
Install CUDA toolkit or skip CUDA experiments:

```python
# In experiment_config.py
IMPLEMENTATIONS = {
    "direct_summation": ["cupy"],  # Remove "cuda"
    "monte_carlo": ["cupy"],
}
```

### "Out of memory"
Reduce chunk_size or test smaller N values:

```python
# In experiment_config.py
N_VALUES = [10000, 100000]  # Skip 1M
```

### Experiment Failed
Check logs:
```bash
tail -f experiments/run_YYYYMMDD_HHMMSS/logs/experiment.log
```

## 📝 Next Steps

1. **Run experiments**: `python run_experiments.py`
2. **Generate plots**: `python generate_plots.py --results experiments/.../results.csv`
3. **Analyze results**: Open the plots and CSV file
4. **Write report**: Use the results and plots in your assignment report

## 🎓 For Your Report

The automation system generates everything you need:

1. **Timing data**: Mean ± std for all methods/implementations
2. **Accuracy data**: Relative errors for validation
3. **Plots**: All required visualizations
4. **CSV**: Easy to import into Excel/LaTeX for tables
5. **JSON**: Detailed metadata for each experiment

Good luck with your assignment! 🚀
