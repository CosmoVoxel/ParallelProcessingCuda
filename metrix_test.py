import subprocess
import os
from pathlib import Path
import itertools

def compile_and_run_with_nvprof(block_size=16, elements_per_thread_x=1, elements_per_thread_y=1, verify=False):
    """Compiles and runs CUDA program with nvprof metrics"""
    
    # Build directory
    build_dir = Path("build")
    build_dir.mkdir(exist_ok=True)
    
    # Results directory
    results_dir = Path("metrix_results")
    results_dir.mkdir(exist_ok=True)
    
    compile_cmd = [
        "nvcc",
        "-std=c++17",
        "-O3",
        "--use_fast_math",
        f"-DELEMENTS_PER_THREAD_X={elements_per_thread_x}",
        f"-DELEMENTS_PER_THREAD_Y={elements_per_thread_y}",
        f"-DVERIFY={'true' if verify else 'false'}",
        f"-DBLOCK_SIZE={block_size}",
        "-o", str(build_dir / "matrixMul"),
        "matrixMul.cu"
    ]

    # Determine executable name based on OS
    file_name = 'matrixMul.exe' if os.name == 'nt' else 'matrixMul'

    # nvprof command with metrics - EXACTLY as you specified!
    nvprof_cmd = [
        "nvprof",
        "--metrics",
        "flop_count_sp,flop_sp_efficiency,achieved_occupancy,shared_load,registers_per_thread,shared_load_transactions,shared_store_transactions,dram_read_throughput,dram_write_throughput",
        "--csv",
        str(build_dir / file_name),
        "-blocksize", str(block_size),
    ]

    # Result filename with parameters
    result_filename = f"elements_per_thread_x_{elements_per_thread_x}_elements_per_thread_y_{elements_per_thread_y}_block_size_{block_size}.csv"
    result_path = results_dir / result_filename

    print(f"Running with command: {' '.join(nvprof_cmd)}")

    try:
        print(f"🔨 Compiling for Block={block_size}, EPT_X={elements_per_thread_x}, EPT_Y={elements_per_thread_y}")
        
        # Compilation
        compile_result = subprocess.run(
            compile_cmd,
            capture_output=True,
            text=True,
            check=True,
            cwd=Path.cwd()
        )
        
        print(f"📊 Running nvprof and saving to {result_filename}")
        
        # Run nvprof and save directly to named file
        with open(result_path, "w", encoding='utf-8') as f:
            nvprof_result = subprocess.run(
                nvprof_cmd,
                stdout=f,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
                timeout=60,
                cwd=Path.cwd()
            )
        
        print(f"✅ Saved: {result_filename}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed for Block={block_size}, EPT_X={elements_per_thread_x}, EPT_Y={elements_per_thread_y}")
        print(f"Error: {e.stderr}")
        return False
    except subprocess.TimeoutExpired:
        print(f"⏰ Timeout for Block={block_size}, EPT_X={elements_per_thread_x}, EPT_Y={elements_per_thread_y}")
        return False

def main():
    """Main function"""
    print("🔥 CUDA Matrix Multiplication nvprof Metrics Collection")
    print("📁 Saving individual CSV files to metrix_results/ folder")
    print("📊 Metrics: flop_count_sp, flop_sp_efficiency, achieved_occupancy, shared_load,")
    print("           registers_per_thread, shared_load_transactions, shared_store_transactions,")
    print("           dram_read_throughput, dram_write_throughput")
    print("=" * 80)
    
    # Test configurations
    block_sizes = [16, 32]
    elements_per_thread_values = [1, 2, 4, 6, 8]
    
    configurations = list(itertools.product(
        block_sizes,
        elements_per_thread_values, 
        elements_per_thread_values
    ))
    
    total_tests = len(configurations)
    successful_tests = 0
    
    print(f"📈 Total configurations to test: {total_tests}")
    print()
    
    # Clean up results directory
    results_dir = Path("metrix_results")
    if results_dir.exists():
        for file in results_dir.glob("*.csv"):
            file.unlink()
        print("🧹 Cleaned up existing CSV files")
    
    # Run tests
    for i, (block_size, ept_x, ept_y) in enumerate(configurations, 1):
        print(f"[{i:2d}/{total_tests}] ", end="")
        
        if compile_and_run_with_nvprof(block_size, ept_x, ept_y, False):
            successful_tests += 1
        
        print()
    
    print("=" * 80)
    print(f"🏁 Completed! {successful_tests}/{total_tests} configurations successful")
    
    if successful_tests > 0:
        results_dir = Path("metrix_results")
        csv_files = list(results_dir.glob("*.csv"))
        print(f"📊 {len(csv_files)} CSV files saved in metrix_results/ folder")
        print()
        print("📂 Generated files:")
        for csv_file in sorted(csv_files):
            file_size = csv_file.stat().st_size
            print(f"   • {csv_file.name} ({file_size} bytes)")
    else:
        print("⚠️  Note: nvprof may not work on modern GPUs (compute capability 7.5+)")
        print("   Consider using NCU (NVIDIA Nsight Compute) instead")

if __name__ == "__main__":
    main()