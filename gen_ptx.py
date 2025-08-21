#!/usr/bin/env python3
"""
Script to generate PTX files from CUDA kernels in the dataset directory.

This script enumerates all .cu files in the dataset directory and compiles them
to PTX format using nvcc with the specified GPU architecture and proper include paths.
"""

import os
import subprocess
import sys
import sysconfig
from pathlib import Path
from typing import List, Tuple


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
        arch: GPU architecture (default: sm_80)
        
    Returns:
        Tuple of (success: bool, error_message: str)
    """
    # Generate output filename
    ptx_filename = cu_file.stem + ".ptx"
    ptx_path = Path(output_dir) / ptx_filename
    
    # Build nvcc command with proper include paths
    cmd = get_ptx_compile_cmd(cu_file, ptx_path, py_include_dir, include_path, deps_path, arch)
    
    print(f"Compiling: {cu_file.name} -> {ptx_filename}")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        # Run nvcc command
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        print(f"✓ Successfully compiled {cu_file.name}")
        if result.stdout:
            print(f"  stdout: {result.stdout.strip()}")
        
        return True, ""
        
    except subprocess.CalledProcessError as e:
        error_msg = f"✗ Failed to compile {cu_file.name}"
        if e.stderr:
            error_msg += f"\n  stderr: {e.stderr.strip()}"
        if e.stdout:
            error_msg += f"\n  stdout: {e.stdout.strip()}"
        
        print(error_msg)
        return False, error_msg
    
    except FileNotFoundError:
        error_msg = "✗ nvcc command not found. Please ensure CUDA toolkit is installed and nvcc is in PATH."
        print(error_msg)
        return False, error_msg


def main():
    """Main function to orchestrate the PTX generation process."""
    
    print("=== CUDA to PTX Compiler ===")
    
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
    
    # Define paths
    dataset_dir = Path(MIRAGE_ROOT) / "dataset"
    output_dir = Path(MIRAGE_ROOT) / "dataset_ptx"
    
    # Convert to absolute paths
    dataset_dir = dataset_dir.resolve()
    output_dir = output_dir.resolve()
    
    print(f"Dataset directory: {dataset_dir}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Find all .cu files
    cu_files = find_cu_files(str(dataset_dir))
    if not cu_files:
        print("No .cu files found in the dataset directory!")
        sys.exit(1)
    
    # Ensure output directory exists
    ensure_output_directory(str(output_dir))
    print()
    
    # Compile each .cu file to PTX
    successful_compilations = 0
    failed_compilations = 0
    errors = []
    
    for cu_file in cu_files:
        success, error_msg = compile_cu_to_ptx(cu_file, str(output_dir), py_include_dir, INCLUDE_PATH, DEPS_PATH)
        if success:
            successful_compilations += 1
        else:
            failed_compilations += 1
            errors.append(error_msg)
        print()  # Add spacing between files
    
    # Print summary
    print("=== Compilation Summary ===")
    print(f"Total files processed: {len(cu_files)}")
    print(f"Successful compilations: {successful_compilations}")
    print(f"Failed compilations: {failed_compilations}")
    
    if errors:
        print("\nErrors encountered:")
        for error in errors:
            print(f"  {error}")
    
    if failed_compilations > 0:
        sys.exit(1)
    else:
        print("\n✓ All files compiled successfully!")


if __name__ == "__main__":
    main()
