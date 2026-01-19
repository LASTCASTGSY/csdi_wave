import matplotlib.pyplot as plt
import numpy as np
import torch
import pickle
import pandas as pd
import os
import glob

def get_quantile(samples, q, dim=1):
    return torch.quantile(samples, q, dim=dim).cpu().numpy()

def plot_wave_results(folder_path=None, sample_idx=0, nsample=100):
    # 1. Automatically find the most recent folder if none provided
    if folder_path is None:
        # Search for folders starting with "wave_imputation" in ./save/
        save_dir = "./save/"
        list_subfolders_with_paths = [f.path for f in os.scandir(save_dir) if f.is_dir()]
        # Filter for wave_imputation folders
        wave_folders = [f for f in list_subfolders_with_paths if "wave_imputation" in f]
        if not wave_folders:
            print("No model folders found in ./save/!")
            return
        # Sort by creation time (newest first)
        folder_path = max(wave_folders, key=os.path.getctime)
        print(f"Auto-detected most recent folder: {folder_path}")
    
    # 2. Load the generated outputs
    path = os.path.join(folder_path, "generated_outputs_nsample10.pk")
    print(f"Loading results from: {path}")
    
    if not os.path.exists(path):
        print(f"Error: File not found at {path}")
        return

    with open(path, 'rb') as f:
        # Load data saved by utils.evaluate
        samples, all_target, all_evalpoint, all_observed, all_observed_time, scaler, mean_scaler = pickle.load(f)

    # 3. Data Preparation & Un-normalization
    # Move to CPU numpy
    all_target_np = all_target.cpu().numpy()
    all_evalpoint_np = all_evalpoint.cpu().numpy()
    all_observed_np = all_observed.cpu().numpy()
    
    # Calculate the mask for "Given" data (Observed - Target to predict)
    all_given_np = all_observed_np - all_evalpoint_np

    # Un-normalize data using the saved scalers
    # scaler = std, mean_scaler = mean
    # Ensure scalers are on CPU
    if torch.is_tensor(scaler): scaler = scaler.cpu().numpy()
    if torch.is_tensor(mean_scaler): mean_scaler = mean_scaler.cpu().numpy()
    
    # Broadcasting un-normalization
    # all_target_np shape: (Batch, Time, Feat)
    # scaler shape: (Feat,)
    all_target_np = all_target_np * scaler + mean_scaler
    
    # samples shape: (Batch, Sample, Time, Feat)
    # We need to broadcast scaler/mean to this shape
    if torch.is_tensor(samples):
        # Un-normalize samples while they are still tensors to avoid massive memory copy
        scaler_t = torch.tensor(scaler, device=samples.device)
        mean_t = torch.tensor(mean_scaler, device=samples.device)
        samples = samples * scaler_t + mean_t
    
    K = samples.shape[-1] # Number of features (should be 9)
    L = samples.shape[-2] # Time length (e.g., 72 or 36)

    # 4. Calculate Quantiles for the green confidence intervals
    qlist = [0.05, 0.25, 0.5, 0.75, 0.95]
    quantiles_imp = []
    
    # We combine the GROUND TRUTH for given points + PREDICTIONS for missing points
    # This makes the line continuous
    for q in qlist:
        # Get quantile from samples
        q_sample = get_quantile(samples, q, dim=1)
        # Combine: If point is given (1), use target. If missing (0), use predicted quantile.
        combined = q_sample * (1 - all_given_np) + all_target_np * all_given_np
        quantiles_imp.append(combined)

    # 5. Plotting
    # Feature names for NDBC data
    feature_names = ['WDIR (deg)', 'WSPD (m/s)', 'WVHT (m)', 'DPD (sec)', 
                     'APD (sec)', 'MWD (deg)', 'PRES (hPa)', 'ATMP (degC)', 'DEWP (degC)']
    
    plt.rcParams["font.size"] = 12
    # 3x3 Grid for 9 features
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(18, 12))
    
    # Check bounds
    if sample_idx >= len(all_target_np):
        print(f"Sample index {sample_idx} out of bounds (max {len(all_target_np)-1}). Resetting to 0.")
        sample_idx = 0

    print(f"Plotting sample index: {sample_idx}")

    for k in range(K):
        row = k // 3
        col = k % 3
        ax = axes[row][col]
        
        # Prepare DataFrames for scattered points
        # Blue Circles = Targets (what we masked and tried to predict)
        df_target = pd.DataFrame({"x": np.arange(0, L), "val": all_target_np[sample_idx, :, k], "y": all_evalpoint_np[sample_idx, :, k]})
        df_target = df_target[df_target.y != 0] # Filter only target points
        
        # Red Crosses = Conditional (what the model saw)
        df_cond = pd.DataFrame({"x": np.arange(0, L), "val": all_target_np[sample_idx, :, k], "y": all_given_np[sample_idx, :, k]})
        df_cond = df_cond[df_cond.y != 0] # Filter only given points

        # Plot Median Prediction (Green Line)
        ax.plot(range(0, L), quantiles_imp[2][sample_idx, :, k], color='g', linestyle='solid', label='CSDI Prediction')
        
        # Plot Confidence Interval (90% interval: 5% to 95%)
        ax.fill_between(range(0, L), 
                        quantiles_imp[0][sample_idx, :, k], 
                        quantiles_imp[4][sample_idx, :, k],
                        color='g', alpha=0.3, label="90% Confidence")

        # Plot Points
        ax.plot(df_target.x, df_target.val, color='b', marker='o', linestyle='None', label='Target (GT)')
        ax.plot(df_cond.x, df_cond.val, color='r', marker='x', linestyle='None', label='Observed')
        
        # Labels and Titles
        if k < len(feature_names):
            ax.set_title(feature_names[k])
        else:
            ax.set_title(f"Feature {k}")
            
        if col == 0:
            ax.set_ylabel('Value')
        if row == 2:
            ax.set_xlabel('Time Step')
            
        ax.grid(True, alpha=0.3)
        
        # Only put legend on the first plot to avoid clutter
        if k == 0:
            ax.legend(loc='upper right', fontsize='small')

    plt.tight_layout()
    
    # Save plot
    save_path = os.path.join(folder_path, f"forecast_plot_sample{sample_idx}.png")
    plt.savefig(save_path)
    print(f"Plot saved to: {save_path}")
    plt.close()

if __name__ == "__main__":
    # You can change sample_idx to view different time windows
    plot_wave_results(sample_idx=0)
    plot_wave_results(sample_idx=5) # Plot another example