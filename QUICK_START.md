# CSDI Wave Height - Quick Start Guide

## What's Included

Complete implementation of wave height imputation and prediction from the paper:
> Si et al., "Improving significant wave height prediction via temporal data imputation", Dynamics of Atmospheres and Oceans, 2025

### All Files Ready to Use

✓ **dataset_wave.py** - NDBC buoy data loaders  
✓ **main_model_wave.py** - CSDI_Wave models for imputation and forecasting  
✓ **exe_wave.py** - Training and evaluation driver  
✓ **config/wave_base.yaml** - Hyperparameter configuration  
✓ **diff_models.py** - Diffusion model architecture (from CSDI)  
✓ **utils.py** - Training/evaluation utilities (from CSDI)  
✓ **preprocess_ndbc_data.py** - Data preprocessing script  
✓ **generate_synthetic_data.py** - Synthetic data generator for testing  
✓ **run_example_workflow.sh** - Complete example workflow  
✓ **README_WAVE.md** - Full documentation  
✓ **IMPLEMENTATION_SUMMARY.md** - Detailed implementation notes  

## Installation

### Prerequisites

```bash
pip install torch numpy pandas pyyaml tqdm
pip install linear-attention-transformer  # For attention layers
```

### Setup

1. Place all files in your working directory:
```
your_project/
├── config/
│   └── wave_base.yaml
├── dataset_wave.py
├── main_model_wave.py
├── exe_wave.py
├── diff_models.py
├── utils.py
├── preprocess_ndbc_data.py
└── generate_synthetic_data.py
```

2. Create data directory:
```bash
mkdir -p data/wave
```

## Quick Test (5 minutes)

Run with synthetic data to verify everything works:

```bash
# 1. Generate synthetic test data
python generate_synthetic_data.py --n_days 10 --output test_data.txt

# 2. Preprocess
python preprocess_ndbc_data.py \
    --input test_data.txt \
    --output ./data/wave \
    --dataset_type NDBC1 \
    --station 42001

# 3. Quick imputation test (reduce epochs for speed)
# Edit config/wave_base.yaml: set epochs: 5 for quick test

# 4. Run imputation
python exe_wave.py \
    --dataset_type NDBC1 \
    --mode imputation \
    --nsample 10 \
    --device cuda:0  # or cpu

# Expected output: Model trains, saves to ./save/wave_imputation_*/
```

## Real Data Usage

### Step 1: Get NDBC Data

Download from: https://www.ndbc.noaa.gov/

Example for station 42001:
- Historical data → Select year/month → Download text file
- Expected format: Space-delimited with columns:
  ```
  #YY MM DD hh mm WDIR WSPD GST WVHT DPD APD MWD PRES ATMP WTMP DEWP ...
  ```

### Step 2: Preprocess

```bash
python preprocess_ndbc_data.py \
    --input /path/to/42001h2017.txt \
    --output ./data/wave \
    --dataset_type NDBC1 \
    --station 42001
```

Creates:
- `NDBC1_42001_processed.pk` - Normalized data and masks
- `NDBC1_42001_meanstd.pk` - Statistics for denormalization

### Step 3: Train Imputation Model

```bash
python exe_wave.py \
    --config wave_base.yaml \
    --dataset_type NDBC1 \
    --station 42001 \
    --mode imputation \
    --device cuda:0 \
    --nsample 100
```

Training time: ~2-4 hours on GPU (200 epochs)  
Output: `./save/wave_imputation_NDBC1_42001_*/`

### Step 4: Train Forecasting Model

```bash
# 1-hour prediction
python exe_wave.py \
    --dataset_type NDBC1 \
    --mode forecasting \
    --forecast_horizon 1 \
    --device cuda:0

# 3-hour prediction
python exe_wave.py \
    --dataset_type NDBC1 \
    --mode forecasting \
    --forecast_horizon 3 \
    --device cuda:0

# 6-hour prediction
python exe_wave.py \
    --dataset_type NDBC1 \
    --mode forecasting \
    --forecast_horizon 6 \
    --device cuda:0
```

Output: `./save/wave_forecasting_NDBC1_42001_Xh_*/`

## Complete Example Workflow

```bash
bash run_example_workflow.sh
```

This runs the entire pipeline:
1. Generates 60 days of synthetic data
2. Preprocesses it
3. Trains imputation model
4. Trains forecasting models (1h and 3h)
5. Saves all results

## Results

Each experiment saves to `./save/wave_*/` with:

- **model.pth** - Trained model weights
- **config.json** - Configuration used
- **generated_outputs_nsample{N}.pk** - Contains:
  - `all_generated_samples`: (n_test, n_samples, L, K) predictions
  - `all_target`: (n_test, L, K) ground truth
  - `all_evalpoint`: (n_test, L, K) evaluation mask
  - `scaler`, `mean_scaler`: For denormalization
  
- **result_nsample{N}.pk** - Contains:
  - RMSE (denormalized)
  - MAE (denormalized)
  - CRPS (probabilistic score)

## Evaluating Results

```python
import pickle
import numpy as np

# Load results
with open('./save/wave_imputation_*/result_nsample100.pk', 'rb') as f:
    rmse, mae, crps = pickle.load(f)

print(f"RMSE: {rmse:.4f} m")  # SWH in meters
print(f"MAE: {mae:.4f} m")
print(f"CRPS: {crps:.4f}")

# Load predictions for visualization
with open('./save/wave_imputation_*/generated_outputs_nsample100.pk', 'rb') as f:
    samples, target, eval_points, obs_points, obs_time, scaler, mean_scaler = pickle.load(f)

# Denormalize predictions
samples_real = samples * scaler.cpu().numpy() + mean_scaler.cpu().numpy()
target_real = target * scaler.cpu().numpy() + mean_scaler.cpu().numpy()

# Extract SWH (feature index 2 for WVHT)
swh_pred = samples_real[:, :, :, 2]  # (n_test, n_samples, L)
swh_true = target_real[:, :, 2]      # (n_test, L)

# Compute median prediction
swh_median = np.median(swh_pred, axis=1)  # (n_test, L)

# Visualize
import matplotlib.pyplot as plt
idx = 0  # First test sample
plt.figure(figsize=(12, 4))
plt.plot(swh_true[idx], 'k-', label='True', linewidth=2)
plt.plot(swh_median[idx], 'r--', label='Predicted (median)', linewidth=2)
plt.fill_between(
    range(len(swh_true[idx])),
    np.quantile(swh_pred[idx], 0.05, axis=0),
    np.quantile(swh_pred[idx], 0.95, axis=0),
    alpha=0.3, label='90% CI'
)
plt.xlabel('Time Step')
plt.ylabel('SWH (m)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.title('Significant Wave Height: Prediction vs Ground Truth')
plt.show()
```

## Common Commands

### Different Dataset Types

```bash
# NDBC1 (10-minute data, 2017-2018)
python exe_wave.py --dataset_type NDBC1 --mode imputation

# NDBC2 (hourly data, 2016)
python exe_wave.py --dataset_type NDBC2 --mode imputation
```

### Different Masking Strategies

```bash
# Mixed (default)
python exe_wave.py --targetstrategy mix

# Random only
python exe_wave.py --targetstrategy random

# Historical pattern only
python exe_wave.py --targetstrategy historical
```

### Load Pre-trained Model

```bash
python exe_wave.py \
    --modelfolder wave_imputation_NDBC1_42001_20250101_120000 \
    --mode imputation \
    --nsample 100
```

### Adjust Configuration

Edit `config/wave_base.yaml`:
```yaml
train:
  epochs: 100        # Reduce for faster training
  batch_size: 32     # Increase if GPU memory allows

diffusion:
  num_steps: 50      # Reduce for faster inference
```

## Troubleshooting

### Out of Memory
- Reduce batch_size in config
- Reduce nsample
- Use shorter forecast horizons

### Slow Training
- Use GPU (cuda:0)
- Reduce epochs for testing
- Reduce num_steps in config

### Poor Results
- Check data preprocessing (missing values, normalization)
- Increase training epochs
- Try different target_strategy
- Verify data format matches NDBC standard

### Import Errors
```bash
pip install linear-attention-transformer
pip install torch numpy pandas pyyaml tqdm
```

## Paper Correspondence

| Paper Section | Implementation |
|--------------|----------------|
| 3.2 Feature Selection | `preprocess_wave_data()` in dataset_wave.py |
| 3.3 Diffusion Model | diff_models.py (CSDI base) |
| 3.4 Imputation/Prediction | CSDI_Wave, CSDI_Wave_Forecasting |
| 3.5 Attention | ResidualBlock in diff_models.py |
| 3.7 Algorithm 1 | `train()` in utils.py |
| 3.7 Algorithm 2 | `impute()` in main_model_wave.py |
| 4.1 NDBC-1 Dataset | Wave_Dataset with dataset_type="NDBC1" |
| 4.1 NDBC-2 Dataset | Wave_Dataset with dataset_type="NDBC2" |
| 4.3 Metrics | `evaluate()` in utils.py |

## Expected Performance

Based on paper Table 4 (NDBC-2):
- **MAE**: ~0.15-0.16 m (CSDI)
- **CRPS**: ~0.15-0.16 (CSDI)

Your results may vary based on:
- Actual missing data patterns
- Training duration
- Random initialization

## Next Steps

1. ✓ Verify installation with synthetic data test
2. ✓ Download real NDBC data
3. ✓ Run imputation experiment
4. ✓ Run forecasting experiment
5. ✓ Analyze and visualize results
6. □ Compare with baseline methods
7. □ Tune hyperparameters for your data
8. □ Deploy for real-time prediction

## Citation

```bibtex
@article{si2025improving,
  title={Improving significant wave height prediction via temporal data imputation},
  author={Si, Jia and Wang, Jie and Deng, Yingjun},
  journal={Dynamics of Atmospheres and Oceans},
  volume={110},
  pages={101549},
  year={2025}
}
```

## Support

- See README_WAVE.md for detailed documentation
- See IMPLEMENTATION_SUMMARY.md for technical details
- Check paper for methodology and experiments

---

**You're ready to go!** All code is complete and tested. Just follow the Quick Test section to verify everything works.
