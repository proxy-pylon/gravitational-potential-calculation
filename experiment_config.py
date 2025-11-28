"""
experiment_config.py - Configuration for automated experiments

This file defines all experimental parameters for systematic testing of:
- Direct summation (CuPy and CUDA)
- Monte Carlo approximation (CuPy and CUDA)
- Deep learning (PyTorch)

Modify these configurations to customize your experiments.
"""

import os
from datetime import datetime

# =============================================================================
# EXPERIMENT PARAMETERS
# =============================================================================

# Particle counts to test (N-body problem size)
N_VALUES = [
    1000,       # 10^3 - very small scale
    10000,      # 10^4 - small scale
    100000,     # 10^5 - medium scale
    1000000,    # 10^6 - large scale
]

# Softening parameter values to test
SOFTENING_VALUES = [
    0.001,      # Very small softening
    0.01,       # Default softening
    0.1,        # Large softening
]

# Monte Carlo sample counts to test
MC_SAMPLE_COUNTS = [
    1000,       # 10^3  - minimal sampling
    10000,      # 10^4  - light sampling
    100000,     # 10^5  - moderate sampling
    # 1000000,    # 10^6  - heavy sampling
    # 10000000,   # 10^7  - very heavy sampling
    # 100000000,  # 10^8  - exhaustive sampling (baseline)
]

# Physical constants
G = 1.0                 # Gravitational constant
SCALE_RADIUS = 1.0      # Plummer scale radius
RANDOM_SEED = 42        # Random seed for reproducibility

# Analytical reference value for Plummer sphere
U_ANALYTICAL = -0.294524

# =============================================================================
# METHOD CONFIGURATIONS
# =============================================================================

# Which methods to run
METHODS = {
    "direct_summation": True,
    "monte_carlo": True,
    "deep_learning": True,
}

# Which implementations to test for each method
IMPLEMENTATIONS = {
    # "direct_summation": ["cupy", "cuda"],
    "direct_summation": ["cuda", "cupy"],
    # "monte_carlo": ["cupy", "cuda"],
    "monte_carlo": ["cuda", "cupy"],
    # "monte_carlo": ["cupy"],
    "deep_learning": ["pytorch"],  # Only PyTorch for DL
}

# =============================================================================
# DEEP LEARNING CONFIGURATION
# =============================================================================

DL_CONFIG = {
    # Training data generation
    "train_n_samples": 10000,       # Number of training samples
    "train_n_bodies": 5000,         # Bodies per training sample
    "train_seed": 42,

    # Model architecture
    "hidden_dims": [128, 256, 128, 64],
    "dropout_rate": 0.1,

    # Training parameters
    "epochs": 100,
    "learning_rate": 0.001,
    "batch_size": 32,
    "val_split": 0.2,
    "early_stop_threshold": 1e-4,
    "patience": 10,
}

# =============================================================================
# TIMING CONFIGURATION
# =============================================================================

TIMING_CONFIG = {
    "warmup_runs": 3,       # Warmup iterations before timing
    "timed_runs": 10,       # Number of timed runs for statistics
}

# =============================================================================
# FILE PATHS AND DIRECTORIES
# =============================================================================

# Base directory for all experiments
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Default output directory (with timestamp)
DEFAULT_OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "experiments",
    f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
)

# Subdirectories within each experiment run
SUBDIRS = {
    "data": "data",                 # Generated Plummer data
    "results": "results",           # JSON result files
    "logs": "logs",                 # Execution logs
    "plots": "plots",               # Generated plots
    "models": "models",             # Trained DL models
}

# =============================================================================
# EXECUTION CONFIGURATION
# =============================================================================

EXECUTION_CONFIG = {
    # Compilation settings for CUDA
    "cuda_compiler": "nvcc",
    "cuda_flags": ["-O3"],          # Optimization flags
    "cuda_build_dir": os.path.join(BASE_DIR, "build"),

    # Python settings
    "python_executable": "python3",

    # Resource limits
    "max_retries": 3,               # Retry failed experiments
    "timeout_seconds": 3600 * 5,        # 5 hour timeout per experiment

    # Logging
    "log_level": "INFO",            # DEBUG, INFO, WARNING, ERROR
    "save_stdout": True,            # Save command output to logs
}

# =============================================================================
# PLOT CONFIGURATION
# =============================================================================

PLOT_CONFIG = {
    "figure_size": (10, 6),
    "dpi": 300,
    "style": "seaborn-v0_8-darkgrid",  # Matplotlib style
    "font_size": 12,
    "save_formats": ["png", "pdf"],     # Output formats

    # Colors for different methods/implementations
    "colors": {
        "direct_cupy": "#1f77b4",
        "direct_cuda": "#ff7f0e",
        "mc_cupy": "#2ca02c",
        "mc_cuda": "#d62728",
        "dl_pytorch": "#9467bd",
    },

    # Plot types to generate
    "plots": [
        "runtime_vs_N",                 # Runtime scaling with N
        "error_vs_N",                   # Accuracy vs N
        "mc_error_vs_K",                # MC error vs sample count
        "speedup_cupy_vs_cuda",         # Implementation comparison
        "error_vs_softening",           # Softening parameter study
    ],
}

# =============================================================================
# VALIDATION THRESHOLDS
# =============================================================================

VALIDATION = {
    "max_relative_error": {
        "direct_summation": 1e-5,       # 0.001%
        "monte_carlo": 0.01,            # 1%
        "deep_learning": 0.01,          # 1%
    },

    # Warn if timing std dev is too high (indicates instability)
    "max_timing_std_ratio": 0.1,       # std / mean < 10%
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_data_filename(n_bodies):
    """Get filename for Plummer data with N bodies."""
    return f"plummer_{n_bodies}.npz"

def get_result_filename(method, implementation, n_bodies, **kwargs):
    """
    Generate result filename based on parameters.

    Args:
        method: Method name (direct_summation, monte_carlo, deep_learning)
        implementation: Implementation type (cupy, cuda, pytorch)
        n_bodies: Number of bodies
        **kwargs: Additional parameters (e.g., n_samples for MC, softening, etc.)

    Returns:
        str: Filename for JSON result
    """
    parts = [
        method.replace("_", ""),
        implementation,
        f"N{n_bodies}"
    ]

    # Add method-specific parameters
    if method == "monte_carlo" and "n_samples" in kwargs:
        parts.append(f"K{kwargs['n_samples']}")

    if "softening" in kwargs:
        eps = kwargs["softening"]
        parts.append(f"eps{eps:.0e}".replace("e-0", "e-"))

    return "_".join(parts) + ".json"

def get_experiment_summary():
    """
    Print a summary of the experiment configuration.
    """
    print("=" * 70)
    print("EXPERIMENT CONFIGURATION SUMMARY")
    print("=" * 70)
    print(f"\nParticle counts (N): {N_VALUES}")
    print(f"Softening values (ε): {SOFTENING_VALUES}")
    print(f"MC sample counts (K): {MC_SAMPLE_COUNTS}")
    print(f"\nMethods enabled:")
    for method, enabled in METHODS.items():
        if enabled:
            impls = ", ".join(IMPLEMENTATIONS[method])
            print(f"  - {method}: {impls}")
    print(f"\nTotal experiments to run:")

    total = 0
    # Direct summation and MC
    for method in ["direct_summation", "monte_carlo"]:
        if METHODS[method]:
            n_impls = len(IMPLEMENTATIONS[method])
            n_configs = len(N_VALUES) * len(SOFTENING_VALUES)
            if method == "monte_carlo":
                n_configs *= len(MC_SAMPLE_COUNTS)
            total += n_impls * n_configs
            print(f"  - {method}: {n_impls * n_configs}")

    # Deep learning
    if METHODS["deep_learning"]:
        dl_total = len(N_VALUES)  # One prediction per N value
        total += dl_total
        print(f"  - deep_learning: {dl_total} (+ 1 training)")

    print(f"\nGRAND TOTAL: {total} experiments")
    print("=" * 70)

if __name__ == "__main__":
    # Print configuration summary when run directly
    get_experiment_summary()
