import argparse
import torch
import datetime
import json
import yaml
import os

from main_model_wave import CSDI_Wave, CSDI_Wave_Forecasting
from dataset_wave import get_dataloader
from utils import train, evaluate

parser = argparse.ArgumentParser(description="CSDI for Wave Height Imputation and Prediction")
parser.add_argument("--config", type=str, default="wave_base.yaml")
parser.add_argument('--device', default='cuda:0', help='Device for training')
parser.add_argument("--modelfolder", type=str, default="", help="Folder with pre-trained model")
parser.add_argument("--nsample", type=int, default=100, help="Number of samples for evaluation")
parser.add_argument("--dataset_type", type=str, default="NDBC1", 
                   choices=["NDBC1", "NDBC2"], help="Dataset type")
parser.add_argument("--station", type=str, default="42001", help="Buoy station ID")
parser.add_argument("--mode", type=str, default="imputation", 
                   choices=["imputation", "forecasting", "both"],
                   help="Task mode: imputation, forecasting, or both")
parser.add_argument("--data_path", type=str, default="./data/wave", 
                   help="Path to wave data directory")
parser.add_argument("--targetstrategy", type=str, default="mix", 
                   choices=["mix", "random", "historical"],
                   help="Target masking strategy for imputation")
parser.add_argument("--unconditional", action="store_true", 
                   help="Use unconditional model")

# Forecasting-specific arguments
parser.add_argument("--forecast_horizon", type=int, default=1, 
                   choices=[1, 3, 6],
                   help="Forecast horizon in hours (1h, 3h, or 6h)")

args = parser.parse_args()
print(args)

# Load config
path = "config/" + args.config
with open(path, "r") as f:
    config = yaml.safe_load(f)

# Update config based on arguments
config["model"]["is_unconditional"] = args.unconditional
config["model"]["target_strategy"] = args.targetstrategy

print(json.dumps(config, indent=4))

# Determine target_dim based on dataset (9 features after removing GST and WTMP)
target_dim = 9

# Create output folder
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def run_imputation():
    """Run imputation task"""
    print("\n" + "="*50)
    print("RUNNING IMPUTATION TASK")
    print("="*50 + "\n")
    
    foldername = (
        f"./save/wave_imputation_{args.dataset_type}_{args.station}_{current_time}/"
    )
    print('Model folder:', foldername)
    os.makedirs(foldername, exist_ok=True)
    
    # Save config
    with open(foldername + "config.json", "w") as f:
        json.dump(config, f, indent=4)
    
    # Get dataloaders
    train_loader, valid_loader, test_loader, scaler, mean_scaler = get_dataloader(
        datatype=args.dataset_type,
        device=args.device,
        batch_size=config["train"]["batch_size"],
        station=args.station,
        data_path=args.data_path,
        task="imputation"
    )
    
    # Initialize model
    model = CSDI_Wave(config, args.device, target_dim).to(args.device)
    
    # Train or load model
    if args.modelfolder == "":
        train(
            model,
            config["train"],
            train_loader,
            valid_loader=valid_loader,
            foldername=foldername,
        )
    else:
        model.load_state_dict(
            torch.load("./save/" + args.modelfolder + "/model.pth")
        )
    
    # Evaluate
    print("\nEvaluating imputation model...")
    evaluate(
        model,
        test_loader,
        nsample=args.nsample,
        scaler=scaler,
        mean_scaler=mean_scaler,
        foldername=foldername,
    )
    
    return foldername


def run_forecasting(imputation_folder=None):
    """Run forecasting task"""
    print("\n" + "="*50)
    print("RUNNING FORECASTING TASK")
    print(f"Forecast Horizon: {args.forecast_horizon}h")
    print("="*50 + "\n")
    
    foldername = (
        f"./save/wave_forecasting_{args.dataset_type}_{args.station}_"
        f"{args.forecast_horizon}h_{current_time}/"
    )
    print('Model folder:', foldername)
    os.makedirs(foldername, exist_ok=True)
    
    # Save config
    with open(foldername + "config.json", "w") as f:
        json.dump(config, f, indent=4)
    
    # Determine history and prediction lengths based on forecast horizon
    # Ratio of historical to prediction is 5:1 as per paper
    if args.dataset_type == "NDBC1":
        # 10-minute intervals
        # 1h = 6 steps, 3h = 18 steps, 6h = 36 steps
        pred_steps = args.forecast_horizon * 6
        history_steps = pred_steps * 5
    elif args.dataset_type == "NDBC2":
        # Hourly intervals
        pred_steps = args.forecast_horizon
        history_steps = pred_steps * 5
    
    print(f"History steps: {history_steps}, Prediction steps: {pred_steps}")
    
    # Get dataloaders
    train_loader, valid_loader, test_loader, scaler, mean_scaler = get_dataloader(
        datatype=args.dataset_type,
        device=args.device,
        batch_size=config["train"]["batch_size"],
        station=args.station,
        data_path=args.data_path,
        task="forecasting",
        history_length=history_steps,
        pred_length=pred_steps
    )
    
    # Initialize forecasting model
    model = CSDI_Wave_Forecasting(config, args.device, target_dim).to(args.device)
    
    # Train or load model
    if args.modelfolder == "":
        train(
            model,
            config["train"],
            train_loader,
            valid_loader=valid_loader,
            foldername=foldername,
        )
    else:
        model.load_state_dict(
            torch.load("./save/" + args.modelfolder + "/model.pth")
        )
    
    # Evaluate
    print(f"\nEvaluating forecasting model for {args.forecast_horizon}h horizon...")
    evaluate(
        model,
        test_loader,
        nsample=args.nsample,
        scaler=scaler,
        mean_scaler=mean_scaler,
        foldername=foldername,
    )
    
    return foldername


# Main execution
if __name__ == "__main__":
    if args.mode == "imputation":
        run_imputation()
    elif args.mode == "forecasting":
        run_forecasting()
    elif args.mode == "both":
        # Run imputation first
        imp_folder = run_imputation()
        print(f"\nImputation completed. Results saved to {imp_folder}")
        
        # Then run forecasting
        # Note: In practice, you would use the imputed data for forecasting
        # This would require saving the imputed data and loading it for forecasting
        forecast_folder = run_forecasting(imputation_folder=imp_folder)
        print(f"\nForecasting completed. Results saved to {forecast_folder}")
    
    print("\n" + "="*50)
    print("EXECUTION COMPLETE")
    print("="*50)
