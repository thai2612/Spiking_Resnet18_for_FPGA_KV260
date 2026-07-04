"""
SNN Benchmark & Accuracy Test for KV260
"""

import time
import numpy as np
import os
from snn_driver_v3 import SNNOverlay


def load_golden_output(filepath):
    """Load golden predicted class from text file.

    Golden files contain:
        logit_0 = 1.027650
        logit_1 = -1.118063
        predicted_class = 0

    Only predicted_class is parsed for comparison (hardware outputs
    int32 logits which cannot be directly compared with float golden logits).
    """
    predicted_class = None
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('predicted_class'):
                try:
                    predicted_class = int(line.split('=')[1].strip())
                except (ValueError, IndexError):
                    pass
    return predicted_class


def run_benchmark(snn, test_images, num_runs=1):
    """Run benchmark on test images."""
    print("\n" + "="*60)
    print(" BENCHMARK")
    print("="*60)

    results = []
    total_time = 0

    for img_path in test_images:
        if not os.path.exists(img_path):
            print(f"SKIP: {img_path} not found")
            continue

        print(f"\nTesting: {os.path.basename(img_path)}")

        # Load image
        image = np.fromfile(img_path, dtype=np.int8).reshape(256, 256, 3)

        # Run inference
        for run in range(num_runs):
            start = time.time()
            logits, pred = snn.predict(image)
            elapsed = time.time() - start
            total_time += elapsed

            results.append({
                'image': os.path.basename(img_path),
                'logits': logits.copy(),
                'pred': pred,
                'time': elapsed
            })

            print(f"  Run {run+1}: {elapsed:.2f}s, pred={pred}, logits={logits}")

    # Summary
    if results:
        times = [r['time'] for r in results]
        print("\n" + "-"*60)
        print(f"Total images: {len(results)}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Average time: {np.mean(times):.2f}s")
        print(f"Min time: {np.min(times):.2f}s")
        print(f"Max time: {np.max(times):.2f}s")
        print(f"FPS: {1/np.mean(times):.4f}")
        print("-"*60)

    return results


def run_accuracy_test(snn, test_images, golden_dir):
    """Run accuracy test comparing with golden outputs."""
    print("\n" + "="*60)
    print(" ACCURACY TEST")
    print("="*60)

    correct = 0
    total = 0

    for img_path in test_images:
        if not os.path.exists(img_path):
            continue

        # Get image index
        basename = os.path.basename(img_path)
        idx = basename.replace('test_input_', '').replace('.bin', '')

        # Load golden output
        golden_path = os.path.join(golden_dir, f'golden_output_{idx}.txt')
        if not os.path.exists(golden_path):
            print(f"SKIP: Golden output not found for {basename}")
            continue

        golden = load_golden_output(golden_path)
        if golden is None:
            print(f"SKIP: Cannot parse golden output for {basename}")
            continue

        # Load and run inference
        image = np.fromfile(img_path, dtype=np.int8).reshape(256, 256, 3)
        logits, pred = snn.predict(image)

        # Compare predicted class
        golden_pred = golden
        match = (pred == golden_pred)

        if match:
            correct += 1
            status = "PASS"
        else:
            status = "FAIL"

        total += 1
        print(f"{basename}: pred={pred}, golden={golden_pred} [{status}]")

    # Summary
    if total > 0:
        accuracy = correct / total * 100
        print("\n" + "-"*60)
        print(f"Accuracy: {correct}/{total} = {accuracy:.1f}%")
        print("-"*60)
    else:
        print("No test cases found!")

    return correct, total


if __name__ == "__main__":
    import sys

    # Default paths
    bitstream = "snn_kv260.bit"
    weights = "network_weights.bin"

    # Find test images
    test_images = []
    for i in range(5):
        path = f"test_input_{i:02d}.bin"
        if os.path.exists(path):
            test_images.append(path)

    if not test_images:
        print("No test images found!")
        print("Looking for: test_input_00.bin, test_input_01.bin, ...")
        sys.exit(1)

    print(f"Found {len(test_images)} test images: {test_images}")

    # Initialize
    print("\nInitializing SNN...")
    snn = SNNOverlay(bitstream)
    snn.load_weights(weights)

    # Run benchmark
    results = run_benchmark(snn, test_images, num_runs=1)

    # Run accuracy test if golden outputs available
    golden_dir = "."  # Same directory
    if any(os.path.exists(f"golden_output_{i:02d}.txt") for i in range(3)):
        run_accuracy_test(snn, test_images, golden_dir)
    else:
        print("\nNo golden outputs found for accuracy test.")
        print("Copy golden_output_XX.txt files to run accuracy test.")
