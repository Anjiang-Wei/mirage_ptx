#!/usr/bin/env python3
"""
Script to generate PTX files from CUDA kernels in the dataset directory for multiple SM architectures.

This script enumerates all .cu files in the dataset directory and compiles them
to PTX format using nvcc for SM80, SM90, and SM100 architectures with separate output folders.
"""

import os
import subprocess
import sys
import sysconfig
import multiprocessing
import time
from pathlib import Path
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed


def get_key_paths():
    """
    Get the key paths for MIRAGE_ROOT, INCLUDE_PATH, and DEPS_PATH.
    This function mirrors the logic from mirage/python/mirage/kernel.py
    """
    # Get the script directory and navigate to mirage root
    script_dir = Path(__file__).parent.resolve()
    root_dir = script_dir  # We're already in the mirage root
    
    # If MIRAGE_ROOT is not set, use the root_dir as MIRAGE_ROOT
    MIRAGE_ROOT = os.environ.get("MIRAGE_ROOT", str(root_dir))
    
    INCLUDE_PATH = ""
    DEPS_PATH = ""
    if os.path.exists(os.path.join(MIRAGE_ROOT, "deps")):
        INCLUDE_PATH = os.path.join(MIRAGE_ROOT, "include")
        DEPS_PATH = os.path.join(MIRAGE_ROOT, "deps")
    else:
        INCLUDE_PATH = os.path.join(MIRAGE_ROOT, "include")
        DEPS_PATH = os.path.join(MIRAGE_ROOT, "include/deps")

    if not os.path.exists(MIRAGE_ROOT):
        raise RuntimeError(f"No MIRAGE_ROOT directory found: {MIRAGE_ROOT}")
    if not os.path.exists(INCLUDE_PATH):
        raise RuntimeError(f"No include directory found: {INCLUDE_PATH}")
    if not os.path.exists(DEPS_PATH):
        raise RuntimeError(f"No deps directory found: {DEPS_PATH}")

    return MIRAGE_ROOT, INCLUDE_PATH, DEPS_PATH


def get_python_include_dir():
    """
    Get the Python include directory path.
    This mirrors the logic from mirage/python/mirage/kernel.py
    """
    # This function was renamed and made public in Python 3.10
    if hasattr(sysconfig, "get_default_scheme"):
        scheme = sysconfig.get_default_scheme()
    else:
        scheme = sysconfig._get_default_scheme()
    # 'posix_local' is a custom scheme on Debian. However, starting Python 3.10, the default install
    # path changes to include 'local'. This change is required to use triton with system-wide python.
    if scheme == "posix_local":
        scheme = "posix_prefix"
    py_include_dir = sysconfig.get_paths(scheme=scheme)["include"]
    return py_include_dir


def find_cu_files(dataset_dir: str) -> List[Path]:
    """
    Find all .cu files in the dataset directory.
    
    Args:
        dataset_dir: Path to the directory containing .cu files
        
    Returns:
        List of Path objects for all .cu files found
    """
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        print(f"Error: Dataset directory {dataset_dir} does not exist!")
        sys.exit(1)
    
    cu_files = list(dataset_path.glob("*.cu"))
    cu_files.sort()  # Sort for consistent ordering
    
    print(f"Found {len(cu_files)} .cu files in {dataset_dir}")
    return cu_files


def ensure_output_directory(output_dir: str) -> None:
    """
    Ensure the output directory exists, create it if necessary.
    
    Args:
        output_dir: Path to the output directory for PTX files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")


def get_ptx_compile_cmd(cu_file: Path, ptx_path: Path, py_include_dir: str, include_path: str, deps_path: str, arch: str = "sm_80") -> List[str]:
    """
    Build the nvcc command for PTX compilation with proper include paths.
    This mirrors the compilation setup from kernel.py but adapted for PTX generation.
    
    Args:
        cu_file: Path to the .cu file to compile
        ptx_path: Path where to save the .ptx file
        py_include_dir: Path to the Python include directory
        include_path: Path to the include directory
        deps_path: Path to the deps directory
        arch: GPU architecture (default: sm_80)
        
    Returns:
        List of command arguments for nvcc
    """
    cmd = [
        "nvcc",
        "-ptx",
        str(cu_file),
        "-O3",
        f"-I{py_include_dir}",
        f"-I{os.path.join(include_path, 'mirage/transpiler/runtime')}",
        f"-I{os.path.join(deps_path, 'cutlass/include')}",
        "-std=c++17",
        "-use_fast_math",
        "--expt-relaxed-constexpr",
        f"-arch={arch}",
        "-o", str(ptx_path),
    ]
    
    return cmd


def compile_cu_to_ptx(cu_file: Path, output_dir: str, py_include_dir: str, include_path: str, deps_path: str, arch: str = "sm_80") -> Tuple[bool, str]:
    """
    Compile a single .cu file to PTX format using nvcc with proper include paths.
    
    Args:
        cu_file: Path to the .cu file to compile
        output_dir: Directory where to save the .ptx file
        py_include_dir: Path to the Python include directory
        include_path: Path to the include directory
        deps_path: Path to the deps directory
        arch: GPU architecture (e.g., sm_80, sm_90, sm_100)
        
    Returns:
        Tuple of (success: bool, error_message: str)
    """
    # Generate output filename
    ptx_filename = cu_file.stem + ".ptx"
    ptx_path = Path(output_dir) / ptx_filename
    
    # Build nvcc command with proper include paths
    cmd = get_ptx_compile_cmd(cu_file, ptx_path, py_include_dir, include_path, deps_path, arch)
    
    # For parallel execution, reduce verbosity - progress is handled by the parallel wrapper
    try:
        # Run nvcc command
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        return True, ""
        
    except subprocess.CalledProcessError as e:
        error_msg = f"Failed to compile ({arch}) {cu_file.name}"
        if e.stderr:
            error_msg += f" - stderr: {e.stderr.strip()}"
        if e.stdout:
            error_msg += f" - stdout: {e.stdout.strip()}"
        
        return False, error_msg
    
    except FileNotFoundError:
        error_msg = f"nvcc command not found when compiling {cu_file.name}. Please ensure CUDA toolkit is installed and nvcc is in PATH."
        return False, error_msg


def compile_files_parallel(cu_files: List[Path], output_dir: str, py_include_dir: str, include_path: str, deps_path: str, arch: str, max_workers: int = None) -> Tuple[int, int, List[str]]:
    """
    Compile multiple .cu files to PTX format in parallel.
    
    Args:
        cu_files: List of .cu files to compile
        output_dir: Directory where to save the .ptx files
        py_include_dir: Path to the Python include directory
        include_path: Path to the include directory
        deps_path: Path to the deps directory
        arch: GPU architecture (e.g., sm_80, sm_90, sm_100)
        max_workers: Maximum number of parallel workers (default: CPU count)
        
    Returns:
        Tuple of (successful_count, failed_count, errors)
    """
    if max_workers is None:
        # Use all available CPU cores for maximum parallelism
        cpu_count = multiprocessing.cpu_count()
        # For nvcc compilation, using all cores is usually optimal
        max_workers = cpu_count
    
    print(f"Starting parallel compilation with {max_workers} workers (using all CPU cores) for {arch.upper()}...")
    print(f"System has {multiprocessing.cpu_count()} CPU cores available")
    print(f"Processing {len(cu_files)} files with {max_workers} parallel workers")
    
    successful_compilations = 0
    failed_compilations = 0
    errors = []
    completed = 0
    
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all compilation jobs
        future_to_file = {
            executor.submit(compile_cu_to_ptx, cu_file, output_dir, py_include_dir, include_path, deps_path, arch): cu_file
            for cu_file in cu_files
        }
        
        # Process completed jobs
        for future in as_completed(future_to_file):
            cu_file = future_to_file[future]
            completed += 1
            
            try:
                success, error_msg = future.result()
                if success:
                    successful_compilations += 1
                    print(f"[{completed:4d}/{len(cu_files)}] ✓ Success ({arch}): {cu_file.name}")
                else:
                    failed_compilations += 1
                    errors.append(error_msg)
                    print(f"[{completed:4d}/{len(cu_files)}] ✗ Failed ({arch}): {cu_file.name}")
                    
            except Exception as e:
                failed_compilations += 1
                error_msg = f"Exception compiling {cu_file.name}: {str(e)}"
                errors.append(error_msg)
                print(f"[{completed:4d}/{len(cu_files)}] ✗ Exception ({arch}): {cu_file.name} - {e}")
            
            # Progress indicator
            if completed % 10 == 0 or completed == len(cu_files):
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (len(cu_files) - completed) / rate if rate > 0 else 0
                print(f"Progress ({arch}): {completed}/{len(cu_files)} ({successful_compilations} successful) "
                      f"- {rate:.1f} files/sec - ETA: {eta/60:.1f}min")
    
    elapsed = time.time() - start_time
    print(f"Parallel compilation completed in {elapsed/60:.1f} minutes ({len(cu_files)/elapsed:.1f} files/sec)")
    
    return successful_compilations, failed_compilations, errors


def main():
    """Main function to orchestrate the PTX generation process for multiple architectures."""
    
    print("=== CUDA to PTX Compiler (Multi-Architecture) ===")
    
    try:
        # Get the key paths for include directories
        MIRAGE_ROOT, INCLUDE_PATH, DEPS_PATH = get_key_paths()
        print(f"MIRAGE_ROOT: {MIRAGE_ROOT}")
        print(f"INCLUDE_PATH: {INCLUDE_PATH}")
        print(f"DEPS_PATH: {DEPS_PATH}")
        
        # Get Python include directory
        py_include_dir = get_python_include_dir()
        print(f"Python include directory: {py_include_dir}")
        
    except RuntimeError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # Define architectures and their corresponding output directories
    architectures = [
        ('sm_80', 'PTX_sm80'),
        ('sm_90', 'PTX_sm90'),
        ('sm_100', 'PTX_sm100')
    ]
    
    # Define paths
    dataset_dir = Path(MIRAGE_ROOT) / "dataset"
    dataset_dir = dataset_dir.resolve()
    
    print(f"Dataset directory: {dataset_dir}")
    print(f"Target architectures: {[arch for arch, _ in architectures]}")
    print(f"Parallel processing: ENABLED using all {multiprocessing.cpu_count()} CPU cores")
    print()
    
    # Find all .cu files
    cu_files = find_cu_files(str(dataset_dir))
    if not cu_files:
        print("No .cu files found in the dataset directory!")
        sys.exit(1)
    
    # Track overall statistics
    total_successful_compilations = 0
    total_failed_compilations = 0
    all_errors = []
    
    # Process each architecture
    for arch, output_folder in architectures:
        print(f"=== Processing {arch.upper()} Architecture ===")
        
        # Define output directory for this architecture
        output_dir = Path(MIRAGE_ROOT) / output_folder
        output_dir = output_dir.resolve()
        
        print(f"Output directory: {output_dir}")
        
        # Ensure output directory exists
        ensure_output_directory(str(output_dir))
        print()
        
        # Compile all .cu files to PTX for this architecture in parallel
        successful_compilations, failed_compilations, errors = compile_files_parallel(
            cu_files, str(output_dir), py_include_dir, INCLUDE_PATH, DEPS_PATH, arch
        )
        
        # Print architecture summary
        print(f"=== {arch.upper()} Architecture Summary ===")
        print(f"Files processed: {len(cu_files)}")
        print(f"Successful compilations: {successful_compilations}")
        print(f"Failed compilations: {failed_compilations}")
        
        if errors:
            print(f"\nErrors encountered for {arch.upper()}:")
            for error in errors:
                print(f"  {error}")
        
        # Update totals
        total_successful_compilations += successful_compilations
        total_failed_compilations += failed_compilations
        all_errors.extend(errors)
        
        print()  # Add spacing between architectures
    
    # Print overall summary
    print("=== Overall Compilation Summary ===")
    print(f"Architectures processed: {len(architectures)}")
    print(f"Total files per architecture: {len(cu_files)}")
    print(f"Total compilations attempted: {len(cu_files) * len(architectures)}")
    print(f"Total successful compilations: {total_successful_compilations}")
    print(f"Total failed compilations: {total_failed_compilations}")
    
    if all_errors:
        print(f"\nTotal errors encountered: {len(all_errors)}")
        print("First few errors:")
        for error in all_errors[:5]:  # Show first 5 errors
            print(f"  {error}")
        if len(all_errors) > 5:
            print(f"  ... and {len(all_errors) - 5} more errors")
    
    if total_failed_compilations > 0:
        sys.exit(1)
    else:
        print("\n✓ All files compiled successfully for all architectures!")


if __name__ == "__main__":
    main()
