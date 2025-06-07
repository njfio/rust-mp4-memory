#!/usr/bin/env python3
"""
Performance test script to demonstrate the improvements in rust_mem_vid.
Creates test files of various sizes to test the performance optimizations.
"""

import os
import time
import subprocess
import tempfile
import shutil
from pathlib import Path

def create_test_file(size_mb, filename):
    """Create a test file of specified size in MB."""
    content = "This is a test line for performance testing. " * 50 + "\n"
    line_size = len(content)
    lines_needed = (size_mb * 1024 * 1024) // line_size
    
    with open(filename, 'w') as f:
        for i in range(lines_needed):
            f.write(f"Line {i}: {content}")
    
    actual_size = os.path.getsize(filename) / (1024 * 1024)
    print(f"Created {filename}: {actual_size:.1f} MB")
    return actual_size

def run_memvid_test(input_file, output_prefix, chunk_size=1000):
    """Run memvid encoding and measure performance."""
    output_video = f"{output_prefix}.mp4"
    output_index = f"{output_prefix}.json"
    
    # Clean up any existing files
    for f in [output_video, output_index]:
        if os.path.exists(f):
            os.remove(f)
    
    print(f"\n🧪 Testing with {input_file} (chunk size: {chunk_size})")
    
    start_time = time.time()
    
    try:
        # Run memvid encode
        cmd = [
            "cargo", "run", "--release", "--",
            "encode",
            "--output", output_video,
            "--index", output_index,
            "--chunk-size", str(chunk_size),
            input_file
        ]
        
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        end_time = time.time()
        duration = end_time - start_time
        
        if result.returncode == 0:
            # Get file sizes
            video_size = os.path.getsize(output_video) if os.path.exists(output_video) else 0
            index_size = os.path.getsize(output_index) if os.path.exists(output_index) else 0
            
            print(f"✅ Success in {duration:.2f} seconds")
            print(f"   Video: {video_size / (1024*1024):.2f} MB")
            print(f"   Index: {index_size / (1024*1024):.2f} MB")
            print(f"   Output preview:")
            for line in result.stdout.split('\n')[-10:]:
                if line.strip():
                    print(f"     {line}")
            
            return {
                'success': True,
                'duration': duration,
                'video_size': video_size,
                'index_size': index_size,
                'output': result.stdout
            }
        else:
            print(f"❌ Failed in {duration:.2f} seconds")
            print(f"Error: {result.stderr}")
            return {
                'success': False,
                'duration': duration,
                'error': result.stderr
            }
    
    except subprocess.TimeoutExpired:
        print("❌ Test timed out after 5 minutes")
        return {
            'success': False,
            'duration': 300,
            'error': 'Timeout'
        }
    
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        return {
            'success': False,
            'duration': 0,
            'error': str(e)
        }

def main():
    """Run performance tests with different file sizes."""
    print("🚀 Rust MemVid Performance Test Suite")
    print("=" * 50)
    
    # Create temporary directory for tests
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Test cases: (size_mb, description)
        test_cases = [
            (0.1, "tiny"),      # 100KB - should use batch processing
            (1, "small"),       # 1MB - should use batch processing  
            (5, "medium"),      # 5MB - should use batch processing
            (10, "large"),      # 10MB - might trigger streaming
        ]
        
        results = []
        
        for size_mb, description in test_cases:
            print(f"\n📁 Creating {description} test file ({size_mb} MB)")
            
            # Create test file
            test_file = temp_path / f"test_{description}.txt"
            actual_size = create_test_file(size_mb, test_file)
            
            # Test with default chunk size
            output_prefix = temp_path / f"output_{description}"
            result = run_memvid_test(str(test_file), str(output_prefix))
            result['file_size_mb'] = actual_size
            result['description'] = description
            results.append(result)
            
            # For larger files, also test with smaller chunk size
            if size_mb >= 5:
                print(f"\n🔧 Testing {description} with smaller chunks")
                output_prefix_small = temp_path / f"output_{description}_small"
                result_small = run_memvid_test(str(test_file), str(output_prefix_small), chunk_size=800)
                result_small['file_size_mb'] = actual_size
                result_small['description'] = f"{description}_small_chunks"
                results.append(result_small)
        
        # Print summary
        print("\n📊 Performance Test Summary")
        print("=" * 50)
        
        for result in results:
            if result['success']:
                throughput = result['file_size_mb'] / result['duration'] if result['duration'] > 0 else 0
                print(f"{result['description']:20} | {result['duration']:6.2f}s | {throughput:6.2f} MB/s | ✅")
            else:
                print(f"{result['description']:20} | {'FAILED':>6} | {'N/A':>6} | ❌")
        
        # Performance insights
        successful_results = [r for r in results if r['success']]
        if successful_results:
            avg_throughput = sum(r['file_size_mb'] / r['duration'] for r in successful_results) / len(successful_results)
            print(f"\n💡 Average throughput: {avg_throughput:.2f} MB/s")
            
            # Check if streaming was used (look for batch processing messages)
            streaming_used = any("streaming" in r.get('output', '').lower() for r in successful_results)
            if streaming_used:
                print("🔄 Streaming processing was automatically activated for large datasets")
            else:
                print("⚡ Batch processing was used for all test cases")

if __name__ == "__main__":
    main()
