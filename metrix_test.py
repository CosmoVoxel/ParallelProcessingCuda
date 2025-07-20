import subprocess
import pandas as pd
import os
from pathlib import Path
import itertools
from tqdm import tqdm

def compile_and_run_with_nvprof(block_size=16, elements_per_thread_x=1, elements_per_thread_y=1, verify=False):
    """Compiles and runs CUDA program with nvprof metrics"""
    
    # Build directory
    build_dir = Path("build")
    build_dir.mkdir(exist_ok=True)
    
    compile_cmd = [
        "nvcc",
        "-std=c++17",
        "-O3",
        "--use_fast_math",
        f"-DELEMENTS_PER_THREAD_X={elements_per_thread_x}",
        f"-DELEMENTS_PER_THREAD_Y={elements_per_thread_y}",
        f"-DVERIFY={'true' if verify else 'false'}",
        "-o", str(build_dir / "matrixMul"),
        "matrixMul.cu"
    ]

    # Determine executable name based on OS
    file_name = 'matrixMul.exe' if os.name == 'nt' else 'matrixMul'


    # nvprof command with metrics
    nvprof_cmd = [
        "nvprof",
        "--metrics",
        "flop_count_sp,flop_sp_efficiency,achieved_occupancy,shared_load,registers_per_thread,shared_load_transactions,shared_store_transactions,dram_read_throughput,dram_write_throughput",
        "--csv",
        str(build_dir / file_name),
    ]

    print(f"Running with command: {' '.join(nvprof_cmd)}")

    try:
        # Compilation
        compile_result = subprocess.run(
            compile_cmd,
            capture_output=True,
            text=True,
            check=True,
            cwd=Path.cwd()
        )
        
        with open("partial_result.csv", "w", encoding='utf-8') as f:
            nvprof_result = subprocess.run(
                nvprof_cmd,
                stdout=f,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
                timeout=60,
                cwd=Path.cwd()
            )
        
        return True, nvprof_result.stderr
        
    except subprocess.CalledProcessError as e:
        return False, f"Process error: {e.stderr}"
    except subprocess.TimeoutExpired:
        return False, "Execution timeout"

def read_partial_results(elements_per_thread_x, elements_per_thread_y, block_size):
    """Reads partial_result.csv and adds configuration info"""
    try:
        # Read the CSV file
        df = pd.read_csv("partial_result.csv")
        
        # Add configuration columns
        df['elements_per_thread_x'] = elements_per_thread_x
        df['elements_per_thread_y'] = elements_per_thread_y
        df['blocksize'] = block_size
        
        # Reorder columns to put config first
        config_cols = ['elements_per_thread_x', 'elements_per_thread_y', 'blocksize']
        other_cols = [col for col in df.columns if col not in config_cols]
        df = df[config_cols + other_cols]
        
        return df
        
    except Exception as e:
        print(f"❌ Error reading partial_result.csv: {e}")
        return None

def append_to_result_csv(df):
    """Appends dataframe to result.csv"""
    try:
        result_file = "result.csv"
        
        # If result.csv exists, append without header, otherwise create with header
        if os.path.exists(result_file):
            df.to_csv(result_file, mode='a', header=False, index=False)
        else:
            df.to_csv(result_file, index=False)
            
        return True
        
    except Exception as e:
        print(f"❌ Error writing to result.csv: {e}")
        return False

def cleanup_partial_result():
    """Removes partial_result.csv"""
    try:
        if os.path.exists("partial_result.csv"):
            os.remove("partial_result.csv")
        return True
    except Exception as e:
        print(f"❌ Error removing partial_result.csv: {e}")
        return False

def run_nvprof_tests():
    """Runs comprehensive nvprof tests"""
    
    # Test configurations
    block_sizes = [16, 32]
    elements_per_thread_values = [1, 2, 4, 6, 8]
    
    total_tests = len(block_sizes) * len(elements_per_thread_values) ** 2
    successful_tests = 0
    
    print("🚀 Starting CUDA Matrix Multiplication nvprof Metrics Tests")
    print("=" * 80)
    print(f"Total configurations to test: {total_tests}")
    print("📊 Metrics: flop_count_sp, flop_sp_efficiency, achieved_occupancy, shared_load,")
    print("           registers_per_thread, shared_load_transactions, shared_store_transactions,")
    print("           dram_read_throughput, dram_write_throughput")
    print()
    
    # Clean up any existing result.csv at start
    if os.path.exists("result.csv"):
        print("🗑️  Removing existing result.csv")
        os.remove("result.csv")
    
    configurations = list(itertools.product(
        block_sizes, 
        elements_per_thread_values, 
        elements_per_thread_values
    ))
    
    for block_size, ept_x, ept_y in tqdm(configurations, desc="Testing configurations"):
        config_name = f"Block={block_size:2d}, EPT_X={ept_x}, EPT_Y={ept_y}"
        
        print(f"\n📋 Testing: {config_name}")
        
        # Run nvprof
        success, stderr = compile_and_run_with_nvprof(
            block_size=block_size,
            elements_per_thread_x=ept_x,
            elements_per_thread_y=ept_y,
            verify=False
        )
        
        if success:
            # Read partial results
            df = read_partial_results(ept_x, ept_y, block_size)
            
            if df is not None and not df.empty:
                # Append to result.csv
                if append_to_result_csv(df):
                    print(f"✅ {config_name} - Data saved to result.csv")
                    successful_tests += 1
                else:
                    print(f"❌ {config_name} - Failed to save data")
            else:
                print(f"❌ {config_name} - No valid data in partial_result.csv")
        else:
            print(f"❌ {config_name} - nvprof failed: {stderr}")
        
        # Clean up partial result file
        cleanup_partial_result()
    
    return successful_tests, total_tests

def analyze_final_results():
    """Analyzes the final result.csv"""
    try:
        if not os.path.exists("result.csv"):
            print("❌ result.csv not found!")
            return
        
        df = pd.read_csv("result.csv")
        
        print(f"\n📊 Final Results Analysis")
        print("=" * 80)
        print(f"📈 Total measurements: {len(df)}")
        print(f"📋 Configurations tested: {len(df.groupby(['elements_per_thread_x', 'elements_per_thread_y', 'blocksize']))}")
        
        if 'Metric Name' in df.columns:
            metrics = df['Metric Name'].unique()
            print(f"📏 Unique metrics: {len(metrics)}")
            for metric in metrics:
                print(f"   • {metric}")
        
        print(f"\n💾 Results saved to: {Path('result.csv').absolute()}")
        print(f"📁 File size: {os.path.getsize('result.csv')} bytes")
        
    except Exception as e:
        print(f"❌ Error analyzing results: {e}")

def main():
    """Main function"""
    print("🔥 CUDA Matrix Multiplication nvprof Metrics Benchmarking Tool")
    print("🎯 Collecting detailed GPU metrics with nvprof\n")
    
    try:
        # Run tests
        successful_tests, total_tests = run_nvprof_tests()
        
        # Analyze final results
        analyze_final_results()
        
        print(f"\n🎉 Benchmarking completed!")
        print(f"📈 Successfully tested {successful_tests}/{total_tests} configurations")
        
        if successful_tests < total_tests:
            print(f"⚠️  {total_tests - successful_tests} configurations failed")
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Benchmarking interrupted by user")
        cleanup_partial_result()
    except Exception as e:
        print(f"\n❌ Error during benchmarking: {e}")
        cleanup_partial_result()

if __name__ == "__main__":
    main()
