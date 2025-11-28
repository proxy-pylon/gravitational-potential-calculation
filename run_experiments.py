"""
run_experiments.py - Automated experiment runner

Systematically runs all configured experiments for gravitational potential calculation.

Usage:
    # Run all experiments
    python run_experiments.py

    # Run specific methods
    python run_experiments.py --methods direct monte_carlo

    # Run with custom output directory
    python run_experiments.py --output_dir experiments/my_run

    # Dry run (show what would be executed)
    python run_experiments.py --dry_run
"""

import os
import sys
import argparse
import subprocess
import json
import csv
import logging
from datetime import datetime
from pathlib import Path
import experiment_config as config

# =============================================================================
# SETUP LOGGING
# =============================================================================

def setup_logging(output_dir, log_level="INFO"):
    """Setup logging configuration."""
    log_dir = os.path.join(output_dir, config.SUBDIRS["logs"])
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, "experiment.log")

    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

    return logging.getLogger(__name__)

# =============================================================================
# DATA GENERATION
# =============================================================================

def generate_plummer_data(n_bodies, output_dir, logger):
    """
    Generate Plummer distribution data for N bodies.

    Args:
        n_bodies: Number of particles
        output_dir: Base output directory
        logger: Logger instance

    Returns:
        str: Path to generated data file
    """
    data_dir = os.path.join(output_dir, config.SUBDIRS["data"])
    os.makedirs(data_dir, exist_ok=True)

    npz_file = os.path.join(data_dir, config.get_data_filename(n_bodies))
    bin_file = os.path.join(data_dir, f"plummer_{n_bodies}.bin")

    # Check if already exists
    if os.path.exists(npz_file) and os.path.exists(bin_file):
        logger.info(f"Data for N={n_bodies} already exists, skipping generation")
        return npz_file

    logger.info(f"Generating Plummer data for N={n_bodies}...")

    cmd = [
        config.EXECUTION_CONFIG["python_executable"],
        "data_gen.py",
        "--n_bodies", str(n_bodies),
        "--scale_radius", str(config.SCALE_RADIUS),
        "--seed", str(config.RANDOM_SEED),
        "--output", npz_file
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=config.EXECUTION_CONFIG["timeout_seconds"],
            check=True
        )
        logger.debug(f"Data generation output: {result.stdout}")
        logger.info(f"✓ Generated data for N={n_bodies}")
        return npz_file
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ Failed to generate data for N={n_bodies}: {e.stderr}")
        raise
    except subprocess.TimeoutExpired:
        logger.error(f"✗ Data generation timed out for N={n_bodies}")
        raise

# =============================================================================
# CUDA COMPILATION
# =============================================================================

def compile_cuda_if_needed(method, logger):
    """
    Compile CUDA programs if they don't exist.

    Args:
        method: Method name (direct_summation or monte_carlo)
        logger: Logger instance

    Returns:
        str: Path to compiled executable
    """
    build_dir = config.EXECUTION_CONFIG["cuda_build_dir"]
    os.makedirs(build_dir, exist_ok=True)

    if method == "direct_summation":
        source = "direct_summation/direct_cuda.cu"
        executable = os.path.join(build_dir, "direct_cuda")
    elif method == "monte_carlo":
        source = "monte_carlo/mc_cuda.cu"
        executable = os.path.join(build_dir, "mc_cuda")
    else:
        raise ValueError(f"Unknown method for CUDA compilation: {method}")

    # Check if already compiled
    if os.path.exists(executable):
        logger.info(f"CUDA executable for {method} already exists")
        return executable

    logger.info(f"Compiling CUDA program for {method}...")

    cmd = [
        config.EXECUTION_CONFIG["cuda_compiler"],
        "-o", executable,
        source
    ] + config.EXECUTION_CONFIG["cuda_flags"]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minutes for compilation
            check=True
        )
        logger.debug(f"Compilation output: {result.stdout}")
        logger.info(f"✓ Compiled {method} CUDA program")
        return executable
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ Failed to compile {method}: {e.stderr}")
        raise

# =============================================================================
# EXPERIMENT EXECUTION
# =============================================================================

def run_direct_summation(implementation, n_bodies, softening, data_file, output_dir, logger):
    """Run direct summation experiment."""
    results_dir = os.path.join(output_dir, config.SUBDIRS["results"])
    os.makedirs(results_dir, exist_ok=True)

    json_file = os.path.join(
        results_dir,
        config.get_result_filename("direct_summation", implementation, n_bodies, softening=softening)
    )

    logger.info(f"Running direct summation ({implementation}) for N={n_bodies}, ε={softening}")

    if implementation == "cupy":
        cmd = [
            config.EXECUTION_CONFIG["python_executable"],
            "direct_summation/direct_cupy.py",
            "--input", data_file,
            "--json_output", json_file,
            "--softening", str(softening)
        ]
    elif implementation == "cuda":
        # Compile if needed
        executable = compile_cuda_if_needed("direct_summation", logger)
        bin_data = data_file.replace(".npz", ".bin")

        cmd = [
            executable,
            "--input", bin_data,
            "--json_output", json_file
        ]
    else:
        raise ValueError(f"Unknown implementation: {implementation}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=config.EXECUTION_CONFIG["timeout_seconds"],
            check=True
        )
        logger.debug(f"Output: {result.stdout}")
        logger.info(f"✓ Completed direct summation ({implementation}) for N={n_bodies}")
        return json_file
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ Failed: {e.stderr}")
        return None
    except subprocess.TimeoutExpired:
        logger.error(f"✗ Timed out")
        return None

def run_monte_carlo(implementation, n_bodies, n_samples, softening, data_file, output_dir, logger):
    """Run Monte Carlo experiment."""
    results_dir = os.path.join(output_dir, config.SUBDIRS["results"])
    os.makedirs(results_dir, exist_ok=True)

    json_file = os.path.join(
        results_dir,
        config.get_result_filename("monte_carlo", implementation, n_bodies,
                                  n_samples=n_samples, softening=softening)
    )

    logger.info(f"Running Monte Carlo ({implementation}) for N={n_bodies}, K={n_samples}, ε={softening}")

    if implementation == "cupy":
        cmd = [
            config.EXECUTION_CONFIG["python_executable"],
            "monte_carlo/mc_cupy.py",
            "--input", data_file,
            "--json_output", json_file,
            "--n_samples", str(n_samples),
            "--softening", str(softening),
            "--seed", str(config.RANDOM_SEED)
        ]
    elif implementation == "cuda":
        executable = compile_cuda_if_needed("monte_carlo", logger)
        bin_data = data_file.replace(".npz", ".bin")

        cmd = [
            executable,
            "--input", bin_data,
            "--json_output", json_file,
            "--n_samples", str(n_samples),
            "--seed", str(config.RANDOM_SEED)
        ]
    else:
        raise ValueError(f"Unknown implementation: {implementation}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=config.EXECUTION_CONFIG["timeout_seconds"],
            check=True
        )
        logger.debug(f"Output: {result.stdout}")
        logger.info(f"✓ Completed Monte Carlo ({implementation}) for N={n_bodies}, K={n_samples}")
        return json_file
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ Failed: {e.stderr}")
        return None
    except subprocess.TimeoutExpired:
        logger.error(f"✗ Timed out")
        return None

def run_deep_learning(n_bodies, data_file, model_file, output_dir, logger):
    """Run deep learning prediction."""
    results_dir = os.path.join(output_dir, config.SUBDIRS["results"])
    os.makedirs(results_dir, exist_ok=True)

    json_file = os.path.join(
        results_dir,
        config.get_result_filename("deep_learning", "pytorch", n_bodies)
    )

    logger.info(f"Running deep learning prediction for N={n_bodies}")

    cmd = [
        config.EXECUTION_CONFIG["python_executable"],
        "deep_learning/mlp_pytorch.py",
        "--predict",
        "--input", data_file,
        "--model", model_file,
        "--json_output", json_file
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=config.EXECUTION_CONFIG["timeout_seconds"],
            check=True
        )
        logger.debug(f"Output: {result.stdout}")
        logger.info(f"✓ Completed deep learning prediction for N={n_bodies}")
        return json_file
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ Failed: {e.stderr}")
        return None
    except subprocess.TimeoutExpired:
        logger.error(f"✗ Timed out")
        return None

# =============================================================================
# RESULT AGGREGATION
# =============================================================================

def aggregate_results(output_dir, logger):
    """
    Aggregate all JSON results into a single CSV file.

    Args:
        output_dir: Experiment output directory
        logger: Logger instance

    Returns:
        str: Path to CSV file
    """
    results_dir = os.path.join(output_dir, config.SUBDIRS["results"])
    csv_file = os.path.join(output_dir, "results.csv")

    logger.info("Aggregating results into CSV...")

    # Collect all JSON files
    json_files = list(Path(results_dir).glob("*.json"))

    if not json_files:
        logger.warning("No JSON result files found")
        return None

    # Read all results
    results = []
    for json_path in json_files:
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                # Flatten nested structure for CSV
                row = {
                    "method": data["method"],
                    "implementation": data["implementation"],
                    "n_bodies": data["parameters"]["n_bodies"],
                    "G": data["parameters"].get("G", config.G),
                    "softening": data["parameters"].get("softening", 0.01),
                    "mean_time_s": data["timing"]["mean_time_seconds"],
                    "std_time_s": data["timing"]["std_time_seconds"],
                    "U_computed": data["results"]["U_computed"],
                    "U_analytical": data["results"]["U_analytical"],
                    "relative_error": data["results"]["relative_error"],
                    "relative_error_pct": data["results"]["relative_error_percent"],
                }

                # Add method-specific fields
                if "n_samples" in data["parameters"]:
                    row["n_samples"] = data["parameters"]["n_samples"]
                if "seed" in data["parameters"]:
                    row["seed"] = data["parameters"]["seed"]

                results.append(row)
        except Exception as e:
            logger.error(f"Error reading {json_path}: {e}")

    # Write CSV
    if results:
        fieldnames = list(results[0].keys())
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        logger.info(f"✓ Aggregated {len(results)} results to {csv_file}")
        return csv_file
    else:
        logger.warning("No results to aggregate")
        return None

# =============================================================================
# MAIN EXPERIMENT ORCHESTRATION
# =============================================================================

def run_all_experiments(output_dir, methods_filter=None, dry_run=False):
    """
    Run all configured experiments.

    Args:
        output_dir: Output directory for results
        methods_filter: List of methods to run (None = all)
        dry_run: If True, only print what would be executed
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Setup logging
    logger = setup_logging(output_dir, config.EXECUTION_CONFIG["log_level"])

    logger.info("=" * 70)
    logger.info("STARTING AUTOMATED EXPERIMENTS")
    logger.info("=" * 70)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Dry run: {dry_run}")

    # Filter methods if specified
    methods_to_run = {}
    for method, enabled in config.METHODS.items():
        if enabled and (methods_filter is None or method in methods_filter):
            methods_to_run[method] = True

    logger.info(f"Methods to run: {list(methods_to_run.keys())}")

    # Generate data files
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 1: GENERATING DATA")
    logger.info("=" * 70)

    data_files = {}
    for n_bodies in config.N_VALUES:
        if not dry_run:
            data_files[n_bodies] = generate_plummer_data(n_bodies, output_dir, logger)
        else:
            logger.info(f"[DRY RUN] Would generate data for N={n_bodies}")

    # Run experiments
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 2: RUNNING EXPERIMENTS")
    logger.info("=" * 70)

    experiment_count = 0
    success_count = 0

    # Direct summation
    if "direct_summation" in methods_to_run:
        for impl in config.IMPLEMENTATIONS["direct_summation"]:
            for n_bodies in config.N_VALUES:
                for softening in config.SOFTENING_VALUES:
                    experiment_count += 1
                    if not dry_run:
                        result = run_direct_summation(
                            impl, n_bodies, softening,
                            data_files[n_bodies], output_dir, logger
                        )
                        if result:
                            success_count += 1
                    else:
                        logger.info(f"[DRY RUN] direct_summation/{impl} N={n_bodies} ε={softening}")

    # Monte Carlo
    if "monte_carlo" in methods_to_run:
        for impl in config.IMPLEMENTATIONS["monte_carlo"]:
            for n_bodies in config.N_VALUES:
                for n_samples in config.MC_SAMPLE_COUNTS:
                    for softening in config.SOFTENING_VALUES:
                        experiment_count += 1
                        if not dry_run:
                            result = run_monte_carlo(
                                impl, n_bodies, n_samples, softening,
                                data_files[n_bodies], output_dir, logger
                            )
                            if result:
                                success_count += 1
                        else:
                            logger.info(f"[DRY RUN] monte_carlo/{impl} N={n_bodies} K={n_samples} ε={softening}")

    # Deep learning (requires trained model)
    if "deep_learning" in methods_to_run:
        model_dir = os.path.join(output_dir, config.SUBDIRS["models"])
        model_file = os.path.join(model_dir, "model.pt")

        # Check if model exists or use pre-trained one
        if not os.path.exists(model_file):
            logger.warning("No trained model found, using existing model from outputs/deep_learning/")
            model_file = "outputs/deep_learning/model.pt"

        for n_bodies in config.N_VALUES:
            experiment_count += 1
            if not dry_run:
                result = run_deep_learning(
                    n_bodies, data_files[n_bodies],
                    model_file, output_dir, logger
                )
                if result:
                    success_count += 1
            else:
                logger.info(f"[DRY RUN] deep_learning/pytorch N={n_bodies}")

    # Aggregate results
    if not dry_run:
        logger.info("\n" + "=" * 70)
        logger.info("PHASE 3: AGGREGATING RESULTS")
        logger.info("=" * 70)
        csv_file = aggregate_results(output_dir, logger)

    # Final summary
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Total experiments: {experiment_count}")
    if not dry_run:
        logger.info(f"Successful: {success_count}")
        logger.info(f"Failed: {experiment_count - success_count}")
        logger.info(f"Success rate: {100*success_count/experiment_count:.1f}%")
        logger.info(f"\nResults saved to: {output_dir}")
        if csv_file:
            logger.info(f"CSV summary: {csv_file}")

    logger.info("=" * 70)

# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run automated experiments for gravitational potential calculation"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=config.DEFAULT_OUTPUT_DIR,
        help="Output directory for results"
    )

    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["direct", "monte_carlo", "deep_learning"],
        default=None,
        help="Methods to run (default: all enabled in config)"
    )

    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Show what would be executed without running"
    )

    args = parser.parse_args()

    # Map short names to full names
    method_map = {
        "direct": "direct_summation",
        "monte_carlo": "monte_carlo",
        "deep_learning": "deep_learning"
    }

    methods_filter = None
    if args.methods:
        methods_filter = [method_map.get(m, m) for m in args.methods]

    # Run experiments
    run_all_experiments(args.output_dir, methods_filter, args.dry_run)

if __name__ == "__main__":
    main()
