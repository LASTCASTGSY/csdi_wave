#!/bin/bash
# Example workflow for CSDI Wave Height Imputation and Prediction
# This script demonstrates the complete pipeline from data generation to evaluation

set -e  # Exit on error

echo "=========================================="
echo "CSDI Wave Height - Example Workflow"
echo "=========================================="
echo ""

# Configuration
DATASET_TYPE="NDBC1"
STATION="42001"
DATA_DIR="./data/wave"
CONFIG="wave_base.yaml"
DEVICE="cuda:0"
N_DAYS=60  # Generate 60 days of synthetic data
NSAMPLE=50  # Use 50 samples for faster testing

echo "Configuration:"
echo "  Dataset type: $DATASET_TYPE"
echo "  Station: $STATION"
echo "  Data directory: $DATA_DIR"
echo "  Device: $DEVICE"
echo "  N samples: $NSAMPLE"
echo ""

# Step 1: Generate synthetic data
echo "=========================================="
echo "Step 1: Generating Synthetic Data"
echo "=========================================="
python generate_synthetic_data.py \
    --n_days $N_DAYS \
    --interval 10 \
    --missing_ratio 0.25 \
    --output synthetic_ndbc_${STATION}.txt \
    --seed 42
echo ""

# Step 2: Preprocess data
echo "=========================================="
echo "Step 2: Preprocessing Data"
echo "=========================================="
python preprocess_ndbc_data.py \
    --input synthetic_ndbc_${STATION}.txt \
    --output $DATA_DIR \
    --dataset_type $DATASET_TYPE \
    --station $STATION
echo ""

# Step 3: Run imputation
echo "=========================================="
echo "Step 3: Training Imputation Model"
echo "=========================================="
python exe_wave.py \
    --config $CONFIG \
    --dataset_type $DATASET_TYPE \
    --station $STATION \
    --mode imputation \
    --device $DEVICE \
    --data_path $DATA_DIR \
    --nsample $NSAMPLE \
    --targetstrategy mix
echo ""

# Step 4: Run 1-hour forecasting
echo "=========================================="
echo "Step 4: Training 1-Hour Forecasting Model"
echo "=========================================="
python exe_wave.py \
    --config $CONFIG \
    --dataset_type $DATASET_TYPE \
    --station $STATION \
    --mode forecasting \
    --forecast_horizon 1 \
    --device $DEVICE \
    --data_path $DATA_DIR \
    --nsample $NSAMPLE
echo ""

# Step 5: Run 3-hour forecasting
echo "=========================================="
echo "Step 5: Training 3-Hour Forecasting Model"
echo "=========================================="
python exe_wave.py \
    --config $CONFIG \
    --dataset_type $DATASET_TYPE \
    --station $STATION \
    --mode forecasting \
    --forecast_horizon 3 \
    --device $DEVICE \
    --data_path $DATA_DIR \
    --nsample $NSAMPLE
echo ""

echo "=========================================="
echo "Workflow Complete!"
echo "=========================================="
echo ""
echo "Results have been saved to ./save/"
echo "Check the following directories for outputs:"
echo "  - ./save/wave_imputation_*/"
echo "  - ./save/wave_forecasting_*_1h_*/"
echo "  - ./save/wave_forecasting_*_3h_*/"
echo ""
echo "Each directory contains:"
echo "  - model.pth: Trained model weights"
echo "  - config.json: Configuration used"
echo "  - generated_outputs_nsample*.pk: Predictions and ground truth"
echo "  - result_nsample*.pk: Evaluation metrics (RMSE, MAE, CRPS)"
