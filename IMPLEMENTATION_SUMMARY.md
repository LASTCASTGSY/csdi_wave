# CSDI Wave Height Implementation - Complete Summary

## Overview

This implementation reproduces the method from:
> J. Si, J. Wang, Y. Deng, "Improving significant wave height prediction via temporal data imputation", *Dynamics of Atmospheres and Oceans*, 2025.

The code extends the official CSDI (Conditional Score-based Diffusion Model) repository to handle NDBC buoy data for significant wave height imputation and prediction.

## Files Created

### Core Implementation Files

1. **dataset_wave.py** (18KB)
   - `Wave_Dataset`: Dataset class for imputation task
   - `Wave_Dataset_Forecasting`: Dataset class for prediction task
   - `get_dataloader()`: Unified dataloader function
   - `preprocess_wave_data()`: Raw data preprocessing function
   - Handles both NDBC1 (10-min) and NDBC2 (hourly) formats
   - Implements feature selection (removes GST and WTMP per paper)
   - Creates train/valid/test splits with proper ratios

2. **main_model_wave.py** (13KB)
   - `CSDI_Wave`: Model for imputation task
   - `CSDI_Wave_Forecasting`: Model for prediction task
   - Both inherit from `CSDI_base` with custom `process_data()` methods
   - Handles tensor permutations to match CSDI conventions

3. **exe_wave.py** (7KB)
   - Main driver script for training and evaluation
   - Supports three modes: `imputation`, `forecasting`, `both`
   - Handles 1h, 3h, 6h forecast horizons
   - Configurable via command-line arguments

4. **config/wave_base.yaml** (1KB)
   - Hyperparameter configuration
   - Based on official CSDI settings
   - Adapted for wave data (9 features)

### Supporting Files

5. **preprocess_ndbc_data.py** (3KB)
   - Standalone preprocessing script
   - Converts raw NDBC files to processed format
   - Handles missing value sentinels (99, 999, etc.)
   - Computes and saves normalization statistics

6. **generate_synthetic_data.py** (8KB)
   - Generates synthetic NDBC-like data for testing
   - Creates realistic wave patterns with correlations
   - Configurable missing data patterns
   - Useful for development and testing

7. **run_example_workflow.sh** (2KB)
   - Complete end-to-end example workflow
   - Demonstrates data generation → preprocessing → training → evaluation
   - Runs both imputation and forecasting tasks

8. **README_WAVE.md** (10KB)
   - Comprehensive documentation
   - Usage instructions and examples
   - Troubleshooting guide
   - Citation information

## Key Features Implemented

### 1. Data Handling

✓ **Feature Selection** (Section 3.2 of paper)
- Pearson correlation-based selection
- Removes GST (high correlation with WSPD) and WTMP (weak correlation with SWH)
- Final 9 features: WDIR, WSPD, WVHT, DPD, APD, MWD, PRES, ATMP, DEWP

✓ **Missing Data Handling**
- Replaces sentinels (99, 999, etc.) with NaN
- Creates binary masks for observed/missing positions
- Normalizes with training set statistics only

✓ **Dataset Splits**
- NDBC1: 2:1:7 ratio (train:valid:test) as per paper
- NDBC2: 7:1:2 ratio (standard temporal split)
- Proper window handling with `eval_length` (72 for NDBC1, 36 for NDBC2)

### 2. Model Architecture

✓ **CSDI Base Model** (Algorithm 1 in paper)
- Conditional score-based diffusion
- Dual attention mechanism (temporal + feature)
- 50 diffusion steps with quadratic schedule
- Target masking strategies: random, historical, mix

✓ **Imputation Module** (Stage 1)
- Trains to impute missing multivariate time series
- Uses masks M_co (condition) and M_ta (target)
- Generates probabilistic distributions for missing values

✓ **Prediction Module** (Stage 2 - Algorithm 2)
- Uses completed data from imputation
- Predicts future SWH values
- Multi-step forecasting with 5:1 history:prediction ratio
- Supports 1h, 3h, 6h horizons for NDBC1

### 3. Training and Evaluation

✓ **Training Loop** (from utils.py)
- Adam optimizer with learning rate scheduling
- Multi-step LR decay at 75% and 90% of epochs
- Validation every 20 epochs
- Best model checkpoint saving

✓ **Evaluation Metrics**
- RMSE: Root Mean Squared Error (denormalized)
- MAE: Mean Absolute Error (denormalized)
- CRPS: Continuous Ranked Probability Score (probabilistic)
- 100 samples generated for uncertainty quantification

## Usage Examples

### Basic Imputation

```bash
# Preprocess data
python preprocess_ndbc_data.py \
    --input raw_data.txt \
    --output ./data/wave \
    --dataset_type NDBC1 \
    --station 42001

# Train imputation model
python exe_wave.py \
    --dataset_type NDBC1 \
    --mode imputation \
    --device cuda:0
```

### Forecasting (1h, 3h, 6h)

```bash
# 1-hour prediction (30 history, 6 pred steps at 10-min intervals)
python exe_wave.py \
    --dataset_type NDBC1 \
    --mode forecasting \
    --forecast_horizon 1

# 3-hour prediction (90 history, 18 pred steps)
python exe_wave.py \
    --dataset_type NDBC1 \
    --mode forecasting \
    --forecast_horizon 3

# 6-hour prediction (180 history, 36 pred steps)
python exe_wave.py \
    --dataset_type NDBC1 \
    --mode forecasting \
    --forecast_horizon 6
```

### Complete Pipeline

```bash
# Run both stages
python exe_wave.py \
    --dataset_type NDBC1 \
    --mode both \
    --forecast_horizon 3
```

### Testing with Synthetic Data

```bash
# Run complete example workflow
bash run_example_workflow.sh
```

## Configuration

The `config/wave_base.yaml` file controls all hyperparameters:

```yaml
train:
  epochs: 200          # Training epochs
  batch_size: 16       # Batch size
  lr: 1.0e-3          # Learning rate

diffusion:
  layers: 4            # Number of residual blocks
  channels: 64         # Hidden dimension
  nheads: 8           # Number of attention heads
  num_steps: 50       # Diffusion steps
  schedule: "quad"    # Noise schedule

model:
  timeemb: 128        # Time embedding dimension
  featureemb: 16      # Feature embedding dimension
  target_strategy: "mix"  # Masking strategy
```

## Implementation Notes

### Dataset Conventions

- **Input format**: (Batch, Length, Features) from dataset
- **Model format**: (Batch, Features, Length) after permutation
- **Output format**: (Batch, Samples, Features, Length) for evaluation

### Masking Strategies

1. **random**: Random element-wise masking (for NDBC2)
2. **historical**: Block-wise masking following observed patterns (for NDBC1)
3. **mix**: Combination of random and historical (default)

### Forecast Horizons (NDBC1, 10-min data)

| Horizon | History Steps | Pred Steps | Total Length |
|---------|--------------|------------|--------------|
| 1h      | 30 (5h)      | 6 (1h)     | 36 (6h)      |
| 3h      | 90 (15h)     | 18 (3h)    | 108 (18h)    |
| 6h      | 180 (30h)    | 36 (6h)    | 216 (36h)    |

All maintain 5:1 ratio as specified in paper.

## Alignment with Paper

### Section 3.2: Feature Selection
✓ Implemented Pearson correlation analysis
✓ Removes GST and WTMP
✓ Uses 9 final features

### Section 3.4: Imputation and Prediction
✓ Unified framework with condition/target masks
✓ Conditional generation p(x_ta | x_co)

### Section 3.5: Attention Mechanism
✓ Dual attention (temporal + feature)
✓ Multi-head attention with residual connections
✓ Layer normalization

### Section 3.7: Algorithms
✓ Algorithm 1: Training procedure implemented in `train()`
✓ Algorithm 2: Sampling procedure implemented in `impute()`

### Section 4: Experiments
✓ NDBC-1 and NDBC-2 dataset support
✓ Evaluation metrics: RMSE, MAE, CRPS
✓ Multiple forecast horizons (1h, 3h, 6h)

## Metrics Correspondence

Paper metrics → Implementation:

- **MAE** (Table 4, 6) → `mae_total / evalpoints_total` in evaluate()
- **MRE** → Not implemented (can compute from MAE/mean)
- **CRPS** (Table 7) → `calc_quantile_CRPS()` in utils.py
- **RMSE** (Table 7) → `sqrt(mse_total / evalpoints_total)`

Results are saved in:
- `result_nsample{N}.pk`: [RMSE, MAE, CRPS]
- `generated_outputs_nsample{N}.pk`: Full predictions

## Differences from Original CSDI

1. **New Classes**:
   - `CSDI_Wave` (replaces PM25/Physio for imputation)
   - `CSDI_Wave_Forecasting` (replaces Forecasting for prediction)

2. **New Datasets**:
   - `Wave_Dataset` (handles NDBC formats and splits)
   - `Wave_Dataset_Forecasting` (handles prediction windows)

3. **New Preprocessing**:
   - NDBC-specific sentinel value handling
   - Feature correlation analysis
   - Proper temporal splits for wave data

4. **Forecast Configurations**:
   - Configurable history/prediction ratios
   - Multiple horizon support (1h, 3h, 6h)

## Dependencies

All dependencies from original CSDI:
- PyTorch
- NumPy
- Pandas
- PyYAML
- tqdm
- linear-attention-transformer (for attention layers)

## Output Structure

```
save/
├── wave_imputation_NDBC1_42001_YYYYMMDD_HHMMSS/
│   ├── model.pth
│   ├── config.json
│   ├── generated_outputs_nsample100.pk
│   └── result_nsample100.pk
└── wave_forecasting_NDBC1_42001_Xh_YYYYMMDD_HHMMSS/
    ├── model.pth
    ├── config.json
    ├── generated_outputs_nsample100.pk
    └── result_nsample100.pk
```

## Testing

Minimal test to verify installation:

```bash
# 1. Generate small synthetic dataset
python generate_synthetic_data.py --n_days 10 --output test.txt

# 2. Preprocess
python preprocess_ndbc_data.py --input test.txt --output ./data/wave

# 3. Quick imputation test (2 epochs)
python exe_wave.py --dataset_type NDBC1 --mode imputation --nsample 10
```

Expected: Model trains without errors, outputs saved to `./save/`.

## Known Limitations

1. **No spatial correlation**: Current implementation uses single-station data. For multi-buoy prediction, additional work needed.

2. **No data augmentation**: Unlike some baselines, no decomposition (EMD, wavelet) preprocessing.

3. **Memory requirements**: Large forecast horizons (6h) require significant GPU memory. Reduce batch_size if OOM.

4. **Inference time**: Diffusion models require many steps (50). Each evaluation takes time.

## Future Extensions

Potential improvements beyond paper:

1. Multi-station spatial modeling
2. Faster sampling (DDIM, DPM-Solver)
3. Conditional forecasting on weather forecasts
4. Ensemble predictions
5. Real-time deployment optimizations

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

---

**Ready to use!** All code is complete and runnable. Simply drop into an existing CSDI repository or use standalone with the provided base files.
