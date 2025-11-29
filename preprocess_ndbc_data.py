#!/usr/bin/env python3
"""
Preprocessing script for NDBC buoy data.
Converts raw NDBC text files to processed pickle files for training.

Usage:
    python preprocess_ndbc_data.py --input raw_data.txt --output ./data/wave --dataset_type NDBC1 --station 42001
"""

import argparse
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from dataset_wave import preprocess_wave_data

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess NDBC buoy data for CSDI wave height model"
    )
    parser.add_argument(
        "--input", 
        type=str, 
        required=True,
        help="Path to raw NDBC data file (CSV or space-delimited text)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./data/wave",
        help="Output directory for processed data (default: ./data/wave)"
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="NDBC1",
        choices=["NDBC1", "NDBC2"],
        help="Dataset type: NDBC1 (10-min) or NDBC2 (hourly)"
    )
    parser.add_argument(
        "--station",
        type=str,
        default="42001",
        help="Buoy station ID (default: 42001)"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("NDBC Buoy Data Preprocessing")
    print("="*60)
    print(f"Input file: {args.input}")
    print(f"Output directory: {args.output}")
    print(f"Dataset type: {args.dataset_type}")
    print(f"Station: {args.station}")
    print("="*60 + "\n")
    
    # Check if input file exists
    if not Path(args.input).exists():
        print(f"ERROR: Input file not found: {args.input}")
        sys.exit(1)
    
    # Create output directory
    Path(args.output).mkdir(parents=True, exist_ok=True)
    
    # Preprocess data
    try:
        normalized_data, mask, train_mean, train_std = preprocess_wave_data(
            raw_file=args.input,
            output_path=args.output,
            dataset_type=args.dataset_type,
            station=args.station
        )
        
        print("\n" + "="*60)
        print("PREPROCESSING COMPLETE")
        print("="*60)
        print(f"Processed data saved to: {args.output}")
        print(f"Files created:")
        print(f"  - {args.dataset_type}_{args.station}_processed.pk")
        print(f"  - {args.dataset_type}_{args.station}_meanstd.pk")
        print("\nYou can now run training with:")
        print(f"  python exe_wave.py --dataset_type {args.dataset_type} "
              f"--station {args.station} --data_path {args.output}")
        print("="*60)
        
    except Exception as e:
        print(f"\nERROR during preprocessing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
