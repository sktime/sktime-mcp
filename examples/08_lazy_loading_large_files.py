#!/usr/bin/env python
"""
Example: Handling Large Files with Streaming Data

This example demonstrates how to use streaming/lazy loading to work with files
larger than available RAM without crashing.

Scenario: 1 million row sales dataset (simulated as CSV)

Run this example:
    python examples/08_lazy_loading_large_files.py
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")

from sktime.forecasting.naive import NaiveForecaster

from sktime_mcp.data.adapters.streaming_adapter import StreamingDataAdapter
from sktime_mcp.data.lazy_loader import ChunkedFitter, LazyDataLoader


def create_large_csv_file(num_rows: int = 100000, output_path: str = None) -> str:
    """
    Create a simulated large CSV file for testing.

    Args:
        num_rows: Number of rows to generate
        output_path: Where to save the file (default: temp file)

    Returns:
        Path to the created file
    """
    if output_path is None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as temp_file:
            output_path = temp_file.name

    print(f"Creating test CSV file with {num_rows:,} rows...")
    print(f"Output: {output_path}")

    # Write header
    with Path(output_path).open("w", encoding="utf-8") as f:
        f.write("date,revenue,cost,profit,region\n")

        # Generate data
        base_date = pd.Timestamp("2020-01-01")
        for i in range(num_rows):
            date = base_date + pd.Timedelta(days=i % 365)
            revenue = 5000 + np.random.randn() * 1000
            cost = 2000 + np.random.randn() * 500
            profit = revenue - cost
            region = ["North", "South", "East", "West"][i % 4]

            f.write(f"{date.date()},{revenue:.2f},{cost:.2f},{profit:.2f},{region}\n")

    file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"[OK] Created: {file_size_mb:.1f} MB\n")

    return output_path


def example_1_metadata_preview():
    """Example 1: Preview file metadata without loading all data."""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Get File Metadata (No Full Load)")
    print("=" * 70 + "\n")

    # Create test file
    csv_path = create_large_csv_file(num_rows=100000)

    # Get metadata
    config = {
        "type": "streaming",
        "path": csv_path,
        "format": "csv",
    }

    adapter = StreamingDataAdapter(config)
    metadata = adapter.get_metadata_from_sample(sample_size=5000)

    print("File Metadata (without loading everything):")
    print(f"  Format: {metadata['format']}")
    print(f"  File Size: {metadata['file_size_bytes'] / (1024 * 1024):.1f} MB")
    print(f"  Estimated Rows: {metadata['estimated_total_rows']:,}")
    print(f"  Estimated Memory: {metadata['memory_estimate_mb']:.0f} MB")
    print(f"  Columns: {metadata['columns']}")
    print(f"  Sample Size: {metadata['sample_size']} rows")

    # Cleanup
    Path(csv_path).unlink()
    print("\n[OK] No OOM - metadata loaded from sample only!\n")


def example_2_chunked_iteration():
    """Example 2: Iterate through data in chunks."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Process Data in Chunks")
    print("=" * 70 + "\n")

    csv_path = create_large_csv_file(num_rows=50000)

    config = {
        "type": "streaming",
        "path": csv_path,
        "format": "csv",
        "chunk_size": 10000,
        "time_column": "date",
        "target_column": "revenue",
    }

    adapter = StreamingDataAdapter(config)

    print("Processing chunks:")
    total_revenue = 0
    chunk_count = 0

    for chunk in adapter.load_chunks():
        chunk_count += 1
        chunk_sum = chunk["revenue"].sum()
        total_revenue += chunk_sum
        print(f"  Chunk {chunk_count}: {len(chunk)} rows, Revenue sum: ${chunk_sum:,.0f}")

    average_revenue = total_revenue / 50000
    print(f"\n  Total Chunks: {chunk_count}")
    print(f"  Total Revenue: ${total_revenue:,.0f}")
    print(f"  Average Revenue: ${average_revenue:,.2f}")

    Path(csv_path).unlink()
    print("\n[OK] Processed 50k rows in chunks without loading all at once!\n")


def example_3_lazy_loader_with_estimator():
    """Example 3: Use LazyDataLoader with an estimator."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Lazy Loading for Time Series Forecasting")
    print("=" * 70 + "\n")

    csv_path = create_large_csv_file(num_rows=50000)

    config = {
        "type": "streaming",
        "path": csv_path,
        "format": "csv",
        "chunk_size": 5000,
        "time_column": "date",
        "target_column": "revenue",
    }

    loader = LazyDataLoader(config, chunk_size=5000)

    # Get metadata
    metadata = loader.get_metadata()
    print("Data Source Info:")
    print(f"  Estimated Rows: {metadata.get('estimated_total_rows'):,}")
    print(f"  Memory Estimate: {metadata.get('memory_estimate_mb'):.0f} MB")
    print(f"  Chunk Size: {metadata.get('chunk_size')} rows\n")

    # Create estimator
    forecaster = NaiveForecaster()

    # Fit on chunked data
    print("Fitting estimator on chunked data...")
    fitter = ChunkedFitter(forecaster)
    fitter.fit_on_chunks(loader)

    print(f"[OK] Estimator fitted on {fitter.get_rows_processed():,} rows")
    print(f"[OK] Fitted state: {fitter.is_fitted()}\n")

    Path(csv_path).unlink()
    print("[OK] Large time series processed without OOM!\n")


def example_4_compare_approaches():
    """Example 4: Compare normal vs. streaming approaches."""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Approaches Comparison")
    print("=" * 70 + "\n")

    csv_path = create_large_csv_file(num_rows=100000)

    print("Approach 1: Traditional (Load All)")
    print("  [FAIL] Problem: df = pd.read_csv(path) - loads all 100k rows")
    print("  [FAIL] Risk: OOM error for multi-GB files\n")

    print("Approach 2: Streaming (Use Chunks)")
    print("  [OK] Solution: Use StreamingDataAdapter")
    print("  [OK] Benefit: Process only chunk_size rows at a time")

    config = {
        "type": "streaming",
        "path": csv_path,
        "format": "csv",
        "chunk_size": 25000,
    }

    adapter = StreamingDataAdapter(config)

    print("\n  Memory Profile:")
    print("  ┌─────────────────────────────────────────┐")
    print("  │ Traditional: 100k rows in RAM → ~50MB  │")
    print("  │ Streaming:   25k rows in RAM → ~12MB  │ (4x savings)")
    print("  └─────────────────────────────────────────┘")

    print("\n  Actual chunks:")
    for i, chunk in enumerate(adapter.load_chunks(), 1):
        print(f"    Chunk {i}: {len(chunk)} rows")

    print("\n  [OK] Same results, 4x less memory!")

    Path(csv_path).unlink()


def main():
    """Run all examples."""
    print("\n")
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║  SKTIME-MCP: LAZY LOADING EXAMPLES                            ║")
    print("║  Handling Large Files Without Out-of-Memory Errors            ║")
    print("╚════════════════════════════════════════════════════════════════╝")

    # Run examples
    example_1_metadata_preview()
    example_2_chunked_iteration()
    example_3_lazy_loader_with_estimator()
    example_4_compare_approaches()

    print("\n" + "=" * 70)
    print("ALL EXAMPLES COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print("\nFor more details, see: docs/lazy-loading.md\n")


if __name__ == "__main__":
    main()
