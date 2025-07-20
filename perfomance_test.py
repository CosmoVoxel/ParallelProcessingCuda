import os
import subprocess
import json
import re
from pathlib import Path
import itertools

def compile_and_run(block_size=16, elements_per_thread_x=1, elements_per_thread_y=1, verify=False):
    """Compiles and runs CUDA program with specified parameters"""
    
    # Build directory
    build_dir = Path("build")
    build_dir.mkdir(exist_ok=True)
    
    # Compilation command
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
    
    file_name = 'matrixMul.exe' if os.name == 'nt' else 'matrixMul'
    
    # Run command
    run_cmd = [
        str(build_dir / file_name),
    ]
    
    try:
        # Compilation
        compile_result = subprocess.run(
            compile_cmd,
            capture_output=True,
            text=True,
            check=True,
            cwd=Path.cwd()
        )
        
        # Execution
        run_result = subprocess.run(
            run_cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=30
        )
        
        return run_result.stdout, run_result.stderr, run_result.returncode
        
    except subprocess.CalledProcessError as e:
        return "", f"Process error: {e.stderr}", e.returncode
    except subprocess.TimeoutExpired:
        return "", "Execution timeout", -1

def parse_performance_output(stdout):
    """Extracts performance data from program output using regex"""
    # Performance line pattern: "Performance= 1234.56 GFlop/s, Time= 1.234 msec, Size= 123456 Ops, WorkgroupSize= 256 threads/block"
    performance_pattern = r'Performance=\s*([\d.]+)\s*GFlop/s,\s*Time=\s*([\d.]+)\s*msec,\s*Size=\s*([\d.]+)\s*Ops,\s*WorkgroupSize=\s*(\d+)\s*threads/block'
    
    match = re.search(performance_pattern, stdout)
    if match:
        return {
            'performance_gflops': float(match.group(1)),
            'time_msec': float(match.group(2)),
            'size_ops': float(match.group(3)),
            'workgroup_size': int(match.group(4))
        }
    return None

def run_performance_tests():
    """Runs comprehensive performance tests"""
    
    # Test configurations
    block_sizes = [16, 32]
    elements_per_thread_values = [1, 2, 4, 6, 8]
    
    results = []
    total_tests = len(block_sizes) * len(elements_per_thread_values) ** 2
    
    print("🚀 Starting CUDA Matrix Multiplication Performance Tests")
    print("=" * 80)
    print(f"Total configurations to test: {total_tests}")
    print()
    
    configurations = list(itertools.product(
        block_sizes, 
        elements_per_thread_values, 
        elements_per_thread_values
    ))
    
    for block_size, ept_x, ept_y in configurations:
        config_name = f"Block={block_size:2d}, EPT_X={ept_x}, EPT_Y={ept_y}"
        
        stdout, stderr, returncode = compile_and_run(
            block_size=block_size,
            elements_per_thread_x=ept_x,
            elements_per_thread_y=ept_y,
            verify=False
        )
        
        if returncode == 0:
            perf_data = parse_performance_output(stdout)
            if perf_data:
                # Validate performance - if > 10000 GFlop/s, likely an error
                if perf_data['performance_gflops'] > 10000:
                    perf_data['performance_gflops'] = -1
                    
                result = {
                    'block_size': block_size,
                    'elements_per_thread_x': ept_x,
                    'elements_per_thread_y': ept_y,
                    'config_name': config_name,
                    **perf_data
                }
                results.append(result)
    
    return results

def save_results(results, filename="performance_results.json"):
    """Saves results to JSON file"""
    output_file = Path(filename)
    
    # Add metadata
    results_with_metadata = {
        'metadata': {
            'total_configurations': len(results),
            'test_date': str(Path(__file__).stat().st_mtime),
            'cuda_program': 'matrixMul.cu'
        },
        'results': results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results_with_metadata, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Results saved to: {output_file.absolute()}")

def analyze_results(results):
    """Analyzes and displays performance results"""
    if not results:
        print("❌ No results to analyze!")
        return
    
    # Filter out error results (-1) for analysis
    valid_results = [r for r in results if r['performance_gflops'] > 0]
    
    print(f"\n📊 Performance Analysis")
    print("=" * 80)
    
    # Top performers
    print("\n🏆 Top 10 Configurations by Performance:")
    print("-" * 80)
    
    sorted_results = sorted(valid_results, key=lambda x: x['performance_gflops'], reverse=True)[:10]
    
    for i, result in enumerate(sorted_results, 1):
        print(f"{i:2d}. {result['config_name']:25s} → "
              f"{result['performance_gflops']:8.2f} GFlop/s "
              f"({result['time_msec']:6.3f} ms)")
    
    # Statistics by block size
    print(f"\n📈 Statistics by Block Size:")
    print("-" * 40)
    
    for block_size in [16, 32]:
        block_results = [r for r in valid_results if r['block_size'] == block_size]
        if block_results:
            performances = [r['performance_gflops'] for r in block_results]
            avg_perf = sum(performances) / len(performances)
            max_perf = max(performances)
            min_perf = min(performances)
            
            print(f"Block Size {block_size:2d}: "
                  f"Avg={avg_perf:6.2f}, "
                  f"Max={max_perf:6.2f}, "
                  f"Min={min_perf:6.2f} GFlop/s "
                  f"({len(block_results)} configs)")
    
    # Best configuration details
    if sorted_results:
        best = sorted_results[0]
        print(f"\n🎯 Best Configuration Details:")
        print("-" * 40)
        print(f"Block Size: {best['block_size']}")
        print(f"Elements per Thread X: {best['elements_per_thread_x']}")
        print(f"Elements per Thread Y: {best['elements_per_thread_y']}")
        print(f"Performance: {best['performance_gflops']:.2f} GFlop/s")
        print(f"Execution Time: {best['time_msec']:.3f} ms")
        print(f"Workgroup Size: {best['workgroup_size']} threads/block")

def main():
    """Main function"""
    print("🔥 CUDA Matrix Multiplication Performance Benchmarking Tool")
    print("🎯 Optimizing for maximum throughput\n")
    
    try:
        # Run tests
        results = run_performance_tests()
        
        # Save results
        save_results(results)
        
        # Analyze results
        analyze_results(results)
        
        print(f"\n🎉 Benchmarking completed successfully!")
        print(f"📈 Tested {len(results)} configurations")
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Benchmarking interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during benchmarking: {e}")

if __name__ == "__main__":
    main()