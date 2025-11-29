# CSDI Wave Height Imputation and Prediction

Implementation of the wave height imputation and prediction method from:

> J. Si, J. Wang, Y. Deng, "Improving significant wave height prediction via temporal data imputation", *Dynamics of Atmospheres and Oceans*, 2025.

This code extends the official CSDI (Conditional Score-based Diffusion Model) repository to handle NDBC buoy data for significant wave height (SWH) imputation and short-term prediction.

## Overview

The method uses a two-stage approach:
1. **Stage 1 - Imputation**: Use CSDI to impute missing values in multivariate buoy time series data
2. **Stage 2 - Prediction**: Use imputed complete data to train a CSDI-based predictor for short-term SWH forecasting (1h, 3h, 6h ahead)

## File Structure

```
.
├── config/
│   └── wave_base.yaml              # Configuration for wave model
├── dataset_wave.py                 # Dataset classes for NDBC buoy data
├── main_model_wave.py              # CSDI_Wave and CSDI_Wave_Forecasting models
├── exe_wave.py                     # Training/evaluation driver script
├── preprocess_ndbc_data.py         # Data preprocessing script
├── diff_models.py                  # Diffusion model architecture (from CSDI)
├── utils.py                        # Training and evaluation utilities (from CSDI)
└── README_WAVE.md                  # This file
```

## Requirements

- Python 3.7+
- PyTorch 1.8+
- NumPy
- Pandas
- PyYAML
- tqdm
- pickle

Additional requirement for attention (from original CSDI):
```bash
pip install linear-attention-transformer
```

## Data Preparation

### NDBC Data Format

The code expects NDBC buoy data in standard text format:
```
YYYY MM DD hh mm  WDIR  WSPD  GST   WVHT   DPD   APD   MWD   PRES   ATMP   WTMP   DEWP
2017 04 27 00 00  136   6.4   7.4   0.57   5.00  3.96  81    1009.7  24.9   25.1   23.0
2017 04 27 00 10  138   6.5   7.3   0.60   5.00  3.98  85    1009.9  24.8   25.1   23.0
...
```

Missing values are encoded as `99.0`, `999`, `999.0`, etc.

### Feature Selection

Following the paper, 9 features are used (GST and WTMP removed):
- WDIR (wind direction)
- WSPD (wind speed)
- WVHT (significant wave height) - **primary target**
- DPD (dominant wave period)
- APD (average wave period)
- MWD (wave direction)
- PRES (sea-level pressure)
- ATMP (air temperature)
- DEWP (dewpoint temperature)

### Preprocessing

Convert raw NDBC data to processed format:

```bash
python preprocess_ndbc_data.py \
    --input /path/to/raw_ndbc_data.txt \
    --output ./data/wave \
    --dataset_type NDBC1 \
    --station 42001
```

Arguments:
- `--input`: Path to raw NDBC data file
- `--output`: Output directory for processed data (default: `./data/wave`)
- `--dataset_type`: `NDBC1` (10-min interval) or `NDBC2` (hourly interval)
- `--station`: Buoy station ID (e.g., `42001`)

This creates:
- `{dataset_type}_{station}_processed.pk`: Normalized data and masks
- `{dataset_type}_{station}_meanstd.pk`: Mean and std for denormalization

## Usage

### 1. Imputation Only

Train CSDI to impute missing values:

```bash
python exe_wave.py \
    --config wave_base.yaml \
    --dataset_type NDBC1 \
    --station 42001 \
    --mode imputation \
    --device cuda:0 \
    --nsample 100
```

### 2. Forecasting Only

Train CSDI for SWH prediction (assumes data is already imputed):

```bash
# 1-hour ahead prediction
python exe_wave.py \
    --config wave_base.yaml \
    --dataset_type NDBC1 \
    --station 42001 \
    --mode forecasting \
    --forecast_horizon 1 \
    --device cuda:0 \
    --nsample 100

# 3-hour ahead prediction
python exe_wave.py \
    --config wave_base.yaml \
    --dataset_type NDBC1 \
    --station 42001 \
    --mode forecasting \
    --forecast_horizon 3 \
    --device cuda:0

# 6-hour ahead prediction
python exe_wave.py \
    --config wave_base.yaml \
    --dataset_type NDBC1 \
    --station 42001 \
    --mode forecasting \
    --forecast_horizon 6 \
    --device cuda:0
```

### 3. Both Imputation and Forecasting

Run the complete two-stage pipeline:

```bash
python exe_wave.py \
    --config wave_base.yaml \
    --dataset_type NDBC1 \
    --station 42001 \
    --mode both \
    --forecast_horizon 3 \
    --device cuda:0
```

### Command-Line Arguments

- `--config`: YAML config file (default: `wave_base.yaml`)
- `--dataset_type`: `NDBC1` or `NDBC2`
- `--station`: Buoy station ID
- `--mode`: `imputation`, `forecasting`, or `both`
- `--forecast_horizon`: For forecasting: `1`, `3`, or `6` hours
- `--device`: Device for training (e.g., `cuda:0` or `cpu`)
- `--nsample`: Number of samples for evaluation (default: 100)
- `--data_path`: Path to processed data directory (default: `./data/wave`)
- `--targetstrategy`: Masking strategy: `mix`, `random`, or `historical` (default: `mix`)
- `--modelfolder`: Folder with pre-trained model (for evaluation only)
- `--unconditional`: Use unconditional model (not recommended)

## Configuration

Edit `config/wave_base.yaml` to adjust hyperparameters:

```yaml
train:
  epochs: 200
  batch_size: 16
  lr: 1.0e-3

diffusion:
  layers: 4
  channels: 64
  nheads: 8
  num_steps: 50
  schedule: "quad"

model:
  timeemb: 128
  featureemb: 16
  target_strategy: "mix"
```

## Datasets

### NDBC-1
- **Period**: 2017-2018 (10-minute intervals)
- **Station**: 42001
- **Sequence length**: 72 time steps (12 hours)
- **Features**: 9 variables
- **Missing pattern**: Block-wise missing (~27.7% - 76.3% missing)
- **Split**: 2:1:7 (train:valid:test)

### NDBC-2
- **Period**: 2016 (hourly intervals)
- **Station**: 42001
- **Sequence length**: 36 time steps (36 hours)
- **Features**: 9 variables
- **Missing pattern**: ~0.8% natural missing + 10% artificial random missing
- **Split**: 7:1:2 (train:valid:test)

## Prediction Settings

For NDBC-1 (10-minute data):
- **1h prediction**: 30 history steps, 6 prediction steps (5:1 ratio)
- **3h prediction**: 90 history steps, 18 prediction steps (5:1 ratio)
- **6h prediction**: 180 history steps, 36 prediction steps (5:1 ratio)

## Evaluation Metrics

The code reports:
- **RMSE**: Root Mean Squared Error (in original units)
- **MAE**: Mean Absolute Error (in original units)
- **CRPS**: Continuous Ranked Probability Score (probabilistic metric)

Results are saved in `./save/wave_*/` folders:
- `model.pth`: Trained model weights
- `generated_outputs_nsample{N}.pk`: Generated samples and ground truth
- `result_nsample{N}.pk`: Evaluation metrics

## Example Workflow

```bash
# 1. Preprocess data
python preprocess_ndbc_data.py \
    --input ./raw_data/ndbc_42001_2017_2018.txt \
    --output ./data/wave \
    --dataset_type NDBC1 \
    --station 42001

# 2. Run imputation
python exe_wave.py \
    --dataset_type NDBC1 \
    --mode imputation \
    --device cuda:0

# 3. Run 3-hour forecasting
python exe_wave.py \
    --dataset_type NDBC1 \
    --mode forecasting \
    --forecast_horizon 3 \
    --device cuda:0
```

## Key Implementation Details

1. **Masking Strategies**:
   - `mix`: Combination of random and historical pattern masking
   - `random`: Random element-wise masking
   - `historical`: Block-wise masking following observed patterns

2. **Attention Mechanism**:
   - Dual attention layers for temporal and feature correlations
   - Transformer-based (not linear attention) as specified in config

3. **Diffusion Process**:
   - 50 diffusion steps (configurable)
   - Quadratic noise schedule
   - Conditional generation based on observed data

4. **Data Normalization**:
   - Z-score normalization using training set statistics
   - Automatic handling of missing values (masked to 0)

## Differences from Original CSDI

This implementation extends the original CSDI codebase with:
- `dataset_wave.py`: Custom dataset for NDBC buoy data
- `main_model_wave.py`: `CSDI_Wave` and `CSDI_Wave_Forecasting` classes
- `exe_wave.py`: Wave-specific training driver
- Support for two-stage imputation→forecasting pipeline
- Feature selection based on Pearson correlation analysis

## Citation

If you use this code, please cite:

```bibtex
@article{si2025improving,
  title={Improving significant wave height prediction via temporal data imputation},
  author={Si, Jia and Wang, Jie and Deng, Yingjun},
  journal={Dynamics of Atmospheres and Oceans},
  volume={110},
  pages={101549},
  year={2025},
  publisher={Elsevier}
}
```

And the original CSDI paper:

```bibtex
@article{tashiro2021csdi,
  title={CSDI: Conditional score-based diffusion models for probabilistic time series imputation},
  author={Tashiro, Yusuke and Song, Jiaming and Song, Yang and Ermon, Stefano},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  pages={24804--24816},
  year={2021}
}
```

## Troubleshooting

### Out of Memory
- Reduce `batch_size` in config
- Reduce `nsample` for evaluation
- Use shorter sequence lengths

### Poor Imputation Results
- Check missing data patterns with visualization
- Try different `target_strategy` values
- Increase training `epochs`
- Adjust `num_steps` in diffusion config

### Data Loading Errors
- Verify NDBC data format matches expected columns
- Check for correct missing value sentinels (99, 999, etc.)
- Ensure preprocessing completed successfully

## Contact

For questions or issues, please refer to the paper or open an issue in the repository.
