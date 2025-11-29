#!/usr/bin/env python3
"""
Generate synthetic NDBC-like data for testing the wave height model.
This creates a synthetic dataset that mimics NDBC buoy data structure.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import argparse


def generate_synthetic_ndbc_data(
    n_days=30,
    interval_minutes=10,
    missing_ratio=0.2,
    output_file="synthetic_ndbc_data.txt",
    seed=42
):
    """
    Generate synthetic NDBC buoy data for testing.
    
    Args:
        n_days: Number of days of data to generate
        interval_minutes: Sampling interval in minutes (10 for NDBC1, 60 for NDBC2)
        missing_ratio: Fraction of data to mark as missing
        output_file: Output filename
        seed: Random seed
    """
    np.random.seed(seed)
    
    # Generate timestamps
    start_date = pd.Timestamp('2017-04-27 00:00:00')
    n_samples = int(n_days * 24 * 60 / interval_minutes)
    timestamps = pd.date_range(start_date, periods=n_samples, freq=f'{interval_minutes}min')
    
    # Generate base signals with some periodicity and noise
    t = np.arange(n_samples)
    
    # Wind direction (degT): 0-360 with some periodicity
    wdir = 180 + 80 * np.sin(2 * np.pi * t / (24 * 60 / interval_minutes)) + \
           20 * np.random.randn(n_samples)
    wdir = np.clip(wdir, 0, 360)
    
    # Wind speed (m/s): typically 0-15 m/s with diurnal variation
    wspd = 7 + 3 * np.sin(2 * np.pi * t / (24 * 60 / interval_minutes)) + \
           2 * np.random.randn(n_samples)
    wspd = np.clip(wspd, 0, 20)
    
    # Gust speed (m/s): slightly higher than wind speed
    gst = wspd + 1 + 0.5 * np.random.randn(n_samples)
    gst = np.clip(gst, wspd, 25)
    
    # Significant wave height (m): correlated with wind speed
    wvht = 0.5 + 0.15 * wspd + 0.3 * np.sin(2 * np.pi * t / (48 * 60 / interval_minutes)) + \
           0.2 * np.random.randn(n_samples)
    wvht = np.clip(wvht, 0.1, 8)
    
    # Dominant wave period (s): 2-12s, weakly correlated with wave height
    dpd = 5 + 2 * np.sqrt(wvht) + 0.5 * np.random.randn(n_samples)
    dpd = np.clip(dpd, 2, 12)
    
    # Average wave period (s): slightly less than dominant
    apd = dpd - 1 + 0.3 * np.random.randn(n_samples)
    apd = np.clip(apd, 2, dpd)
    
    # Mean wave direction (degT): similar to wind direction with offset
    mwd = wdir + 20 + 30 * np.random.randn(n_samples)
    mwd = np.clip(mwd, 0, 360)
    
    # Sea level pressure (hPa): 980-1040 hPa with slow variation
    pres = 1013 + 10 * np.sin(2 * np.pi * t / (7 * 24 * 60 / interval_minutes)) + \
           3 * np.random.randn(n_samples)
    pres = np.clip(pres, 980, 1040)
    
    # Air temperature (°C): 15-30°C with diurnal variation
    atmp = 23 + 4 * np.sin(2 * np.pi * t / (24 * 60 / interval_minutes)) + \
           1.5 * np.random.randn(n_samples)
    atmp = np.clip(atmp, 10, 35)
    
    # Sea surface temperature (°C): similar to air temp but more stable
    wtmp = 25 + 2 * np.sin(2 * np.pi * t / (30 * 24 * 60 / interval_minutes)) + \
           0.5 * np.random.randn(n_samples)
    wtmp = np.clip(wtmp, 20, 32)
    
    # Dewpoint temperature (°C): lower than air temp
    dewp = atmp - 3 - 2 * np.random.rand(n_samples)
    dewp = np.clip(dewp, 5, atmp - 1)
    
    # Create DataFrame
    df = pd.DataFrame({
        '#YY': timestamps.year,
        'MM': timestamps.month,
        'DD': timestamps.day,
        'hh': timestamps.hour,
        'mm': timestamps.minute,
        'WDIR': wdir,
        'WSPD': wspd,
        'GST': gst,
        'WVHT': wvht,
        'DPD': dpd,
        'APD': apd,
        'MWD': mwd,
        'PRES': pres,
        'ATMP': atmp,
        'WTMP': wtmp,
        'DEWP': dewp
    })
    
    # Introduce missing values (replace with 999.0)
    if missing_ratio > 0:
        n_missing = int(n_samples * missing_ratio)
        # Randomly select rows to mark as missing (block missing pattern)
        missing_indices = np.random.choice(n_samples, n_missing, replace=False)
        
        # Mark some variables as missing in these rows
        for idx in missing_indices:
            # Randomly choose which variables to mark as missing
            vars_to_miss = np.random.choice(
                ['WVHT', 'DPD', 'APD', 'MWD'], 
                size=np.random.randint(1, 4),
                replace=False
            )
            for var in vars_to_miss:
                df.loc[idx, var] = 999.0
    
    # Format and save
    # Convert to proper format
    df['#YY'] = df['#YY'].astype(int)
    df['MM'] = df['MM'].apply(lambda x: f'{x:02d}')
    df['DD'] = df['DD'].apply(lambda x: f'{x:02d}')
    df['hh'] = df['hh'].apply(lambda x: f'{x:02d}')
    df['mm'] = df['mm'].apply(lambda x: f'{x:02d}')
    
    # Round numerical values
    for col in ['WDIR', 'WSPD', 'GST', 'WVHT', 'DPD', 'APD', 'MWD', 'PRES', 'ATMP', 'WTMP', 'DEWP']:
        df[col] = df[col].round(2)
    
    # Save to file
    df.to_csv(output_file, sep=' ', index=False, float_format='%.2f')
    
    print(f"Generated synthetic NDBC data:")
    print(f"  - Output file: {output_file}")
    print(f"  - Number of samples: {n_samples}")
    print(f"  - Time period: {n_days} days")
    print(f"  - Sampling interval: {interval_minutes} minutes")
    print(f"  - Missing ratio: {missing_ratio:.1%}")
    print(f"  - Actual missing values: {(df == 999.0).sum().sum()}")
    
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic NDBC data for testing")
    parser.add_argument("--n_days", type=int, default=30, help="Number of days of data")
    parser.add_argument("--interval", type=int, default=10, 
                       help="Sampling interval in minutes (10 or 60)")
    parser.add_argument("--missing_ratio", type=float, default=0.2, 
                       help="Fraction of missing data (0.0 to 1.0)")
    parser.add_argument("--output", type=str, default="synthetic_ndbc_data.txt",
                       help="Output filename")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    generate_synthetic_ndbc_data(
        n_days=args.n_days,
        interval_minutes=args.interval,
        missing_ratio=args.missing_ratio,
        output_file=args.output,
        seed=args.seed
    )
