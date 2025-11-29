import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from pathlib import Path


class Wave_Dataset(Dataset):
    """
    Dataset class for NDBC buoy wave data.
    Supports two dataset types:
    - NDBC1: 10-min interval data (2017-2018) with 9 features, eval_length=72
    - NDBC2: 1-hour interval data (2016) with 9 features, eval_length=36
    """
    def __init__(self, dataset_type="NDBC1", mode="train", validindex=0, 
                 station="42001", data_path="./data/wave"):
        self.dataset_type = dataset_type
        self.mode = mode
        self.validindex = validindex
        self.station = station
        self.data_path = Path(data_path)
        
        # Set eval_length based on dataset type
        if dataset_type == "NDBC1":
            self.eval_length = 72  # 72 time steps at 10-min interval = 12 hours
        elif dataset_type == "NDBC2":
            self.eval_length = 36  # 36 time steps at 1-hour interval = 36 hours
        else:
            raise ValueError(f"Unknown dataset_type: {dataset_type}")
        
        # Feature selection: 9 features after removing GST and WTMP
        # WDIR, WSPD, WVHT, DPD, APD, MWD, PRES, ATMP, DEWP
        self.feature_names = ['WDIR', 'WSPD', 'WVHT', 'DPD', 'APD', 'MWD', 'PRES', 'ATMP', 'DEWP']
        self.target_dim = len(self.feature_names)
        
        # Load normalization statistics
        meanstd_path = self.data_path / f"{dataset_type}_{station}_meanstd.pk"
        if meanstd_path.exists():
            with open(meanstd_path, "rb") as f:
                self.train_mean, self.train_std = pickle.load(f)
        else:
            # Will be created during preprocessing
            self.train_mean = None
            self.train_std = None
        
        # Load and preprocess data
        self._load_data()
        
        # Create train/val/test splits
        self._create_splits()
    
    def _load_data(self):
        """Load and preprocess NDBC buoy data"""
        data_file = self.data_path / f"{self.dataset_type}_{self.station}_processed.pk"
        
        if data_file.exists():
            with open(data_file, "rb") as f:
                data_dict = pickle.load(f)
                self.observed_data = data_dict['observed_data']
                self.observed_mask = data_dict['observed_mask']
                if self.train_mean is None:
                    self.train_mean = data_dict['train_mean']
                    self.train_std = data_dict['train_std']
        else:
            raise FileNotFoundError(
                f"Processed data file not found: {data_file}\n"
                f"Please run preprocess_wave_data() first to create this file."
            )
    
    def _create_splits(self):
        """Create train/valid/test indices"""
        total_length = len(self.observed_data)
        
        if self.dataset_type == "NDBC1":
            # For NDBC1, use 2:1:7 ratio for train:valid:test
            train_ratio = 0.2
            valid_ratio = 0.1
            test_ratio = 0.7
            
            train_end = int(total_length * train_ratio)
            valid_end = train_end + int(total_length * valid_ratio)
            
            if self.mode == "train":
                start = 0
                end = train_end - self.eval_length + 1
                self.use_index = np.arange(start, end, 1)
            elif self.mode == "valid":
                start = train_end
                end = valid_end - self.eval_length + 1
                self.use_index = np.arange(start, end, 1)
            elif self.mode == "test":
                start = valid_end
                end = total_length - self.eval_length + 1
                # Use non-overlapping windows for test
                self.use_index = np.arange(start, end, self.eval_length)
        
        elif self.dataset_type == "NDBC2":
            # For NDBC2, similar to PM2.5 dataset but simpler
            # Use month-based split is not applicable here, use simple temporal split
            train_ratio = 0.7
            valid_ratio = 0.1
            
            train_end = int(total_length * train_ratio)
            valid_end = train_end + int(total_length * valid_ratio)
            
            if self.mode == "train":
                start = 0
                end = train_end - self.eval_length + 1
                self.use_index = np.arange(start, end, 1)
            elif self.mode == "valid":
                start = train_end
                end = valid_end - self.eval_length + 1
                self.use_index = np.arange(start, end, 1)
            elif self.mode == "test":
                start = valid_end
                end = total_length - self.eval_length + 1
                self.use_index = np.arange(start, end, self.eval_length)
        
        # Initialize cut_length (for avoiding double evaluation at boundaries)
        self.cut_length = [0] * len(self.use_index)
        
        # For test mode, handle last window if needed
        if self.mode == "test" and len(self.observed_data) % self.eval_length != 0:
            remainder = len(self.observed_data) % self.eval_length
            if remainder > 0 and self.use_index[-1] + self.eval_length > len(self.observed_data):
                self.cut_length[-1] = self.eval_length - remainder
    
    def __getitem__(self, org_index):
        index = self.use_index[org_index]
        
        # Extract window
        obs_data = self.observed_data[index:index + self.eval_length]
        obs_mask = self.observed_mask[index:index + self.eval_length]
        
        # Create gt_mask based on mode and dataset type
        if self.mode == "train":
            # During training, we artificially mask some observed values
            # For NDBC1: use test pattern (block missing)
            # For NDBC2: use random missing
            gt_mask = obs_mask.copy()
        else:
            # During validation/test, use actual observed mask
            gt_mask = obs_mask.copy()
        
        # Create timepoints
        timepoints = np.arange(self.eval_length) * 1.0
        
        s = {
            'observed_data': obs_data,  # (L, K)
            'observed_mask': obs_mask,  # (L, K)
            'gt_mask': gt_mask,  # (L, K)
            'timepoints': timepoints,  # (L,)
            'cut_length': self.cut_length[org_index],
        }
        
        # Add hist_mask for pattern-based training (similar to PM2.5)
        if self.mode == "train":
            # Use same window for hist_mask (simplified version)
            s['hist_mask'] = obs_mask.copy()
        else:
            s['hist_mask'] = obs_mask.copy()
        
        return s
    
    def __len__(self):
        return len(self.use_index)


class Wave_Dataset_Forecasting(Dataset):
    """
    Dataset class for wave height forecasting.
    Uses completed/imputed data to predict future SWH values.
    """
    def __init__(self, dataset_type="NDBC1", mode="train", 
                 station="42001", data_path="./data/wave",
                 history_length=None, pred_length=None):
        self.dataset_type = dataset_type
        self.mode = mode
        self.station = station
        self.data_path = Path(data_path)
        
        # Set default history and prediction lengths based on dataset type
        if dataset_type == "NDBC1":
            # For 10-min data: default to 5:1 ratio
            # 1h prediction: 30 history, 6 pred (5:1)
            # 3h prediction: 90 history, 18 pred (5:1)
            # 6h prediction: 180 history, 36 pred (5:1)
            if history_length is None:
                self.history_length = 30  # Default to 1h prediction
            else:
                self.history_length = history_length
            
            if pred_length is None:
                self.pred_length = 6  # 1h ahead at 10-min intervals
            else:
                self.pred_length = pred_length
        elif dataset_type == "NDBC2":
            # For hourly data
            if history_length is None:
                self.history_length = 168  # 1 week
            else:
                self.history_length = history_length
            
            if pred_length is None:
                self.pred_length = 24  # 1 day ahead
            else:
                self.pred_length = pred_length
        
        self.seq_length = self.history_length + self.pred_length
        
        # Load processed/imputed data
        self._load_data()
        
        # Create splits
        self._create_splits()
    
    def _load_data(self):
        """Load imputed/completed wave data"""
        # Try to load imputed data first, otherwise use processed data
        imputed_file = self.data_path / f"{self.dataset_type}_{self.station}_imputed.pk"
        processed_file = self.data_path / f"{self.dataset_type}_{self.station}_processed.pk"
        
        if imputed_file.exists():
            with open(imputed_file, "rb") as f:
                data_dict = pickle.load(f)
                self.main_data = data_dict['imputed_data']
                self.mask_data = np.ones_like(self.main_data)  # All imputed
                self.mean_data = data_dict['train_mean']
                self.std_data = data_dict['train_std']
        elif processed_file.exists():
            with open(processed_file, "rb") as f:
                data_dict = pickle.load(f)
                # Use observed data with zeros for missing
                obs_data = data_dict['observed_data']
                obs_mask = data_dict['observed_mask']
                # Fill missing with mean (simple approach)
                self.main_data = obs_data.copy()
                for i in range(obs_data.shape[1]):
                    mean_val = np.nanmean(obs_data[:, i])
                    self.main_data[obs_mask[:, i] == 0, i] = mean_val
                self.mask_data = obs_mask
                self.mean_data = data_dict['train_mean']
                self.std_data = data_dict['train_std']
        else:
            raise FileNotFoundError(f"No data found at {self.data_path}")
        
        # Normalize data
        self.main_data = (self.main_data - self.mean_data) / self.std_data
    
    def _create_splits(self):
        """Create train/valid/test indices for forecasting"""
        total_length = len(self.main_data)
        
        # Reserve last portion for validation and test
        # Following Nguyen & Quanz (2021): last 320 sequences for validation
        if self.mode == "train":
            start = 0
            end = total_length - self.seq_length - 320 * self.pred_length + 1
            self.use_index = np.arange(start, end, 1)
        elif self.mode == "valid":
            start = total_length - 320 * self.pred_length
            end = total_length - self.seq_length + 1
            self.use_index = np.arange(start, end, self.pred_length)
        elif self.mode == "test":
            start = total_length - 320 * self.pred_length
            end = total_length - self.seq_length + 1
            self.use_index = np.arange(start, end, self.pred_length)
    
    def __getitem__(self, org_index):
        index = self.use_index[org_index]
        
        # Get sequence
        data = self.main_data[index:index + self.seq_length]
        mask = self.mask_data[index:index + self.seq_length]
        
        # Create target mask: future values (last pred_length steps) are targets
        target_mask = mask.copy()
        target_mask[-self.pred_length:] = 0.0  # Mark future as targets
        
        s = {
            'observed_data': data,  # (L, K)
            'observed_mask': mask,  # (L, K)
            'gt_mask': target_mask,  # (L, K)
            'timepoints': np.arange(self.seq_length) * 1.0,
            'feature_id': np.arange(data.shape[1]) * 1.0,
        }
        
        return s
    
    def __len__(self):
        return len(self.use_index)


def get_dataloader(datatype="NDBC1", device='cuda:0', batch_size=16, 
                   station="42001", data_path="./data/wave",
                   task="imputation", **kwargs):
    """
    Get data loaders for wave dataset.
    
    Args:
        datatype: "NDBC1" or "NDBC2"
        device: torch device
        batch_size: batch size
        station: buoy station ID
        data_path: path to data directory
        task: "imputation" or "forecasting"
        **kwargs: additional arguments for forecasting (history_length, pred_length)
    
    Returns:
        train_loader, valid_loader, test_loader, scaler, mean_scaler
    """
    if task == "imputation":
        dataset = Wave_Dataset(
            dataset_type=datatype, 
            mode='train',
            station=station,
            data_path=data_path
        )
        train_loader = DataLoader(
            dataset, batch_size=batch_size, num_workers=1, shuffle=True
        )
        
        valid_dataset = Wave_Dataset(
            dataset_type=datatype,
            mode='valid',
            station=station,
            data_path=data_path
        )
        valid_loader = DataLoader(
            valid_dataset, batch_size=batch_size, num_workers=1, shuffle=False
        )
        
        test_dataset = Wave_Dataset(
            dataset_type=datatype,
            mode='test',
            station=station,
            data_path=data_path
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, num_workers=1, shuffle=False
        )
    
    elif task == "forecasting":
        dataset = Wave_Dataset_Forecasting(
            dataset_type=datatype,
            mode='train',
            station=station,
            data_path=data_path,
            history_length=kwargs.get('history_length'),
            pred_length=kwargs.get('pred_length')
        )
        train_loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )
        
        valid_dataset = Wave_Dataset_Forecasting(
            dataset_type=datatype,
            mode='valid',
            station=station,
            data_path=data_path,
            history_length=kwargs.get('history_length'),
            pred_length=kwargs.get('pred_length')
        )
        valid_loader = DataLoader(
            valid_dataset, batch_size=batch_size, shuffle=False
        )
        
        test_dataset = Wave_Dataset_Forecasting(
            dataset_type=datatype,
            mode='test',
            station=station,
            data_path=data_path,
            history_length=kwargs.get('history_length'),
            pred_length=kwargs.get('pred_length')
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False
        )
    
    else:
        raise ValueError(f"Unknown task: {task}")
    
    scaler = torch.from_numpy(dataset.train_std).to(device).float()
    mean_scaler = torch.from_numpy(dataset.train_mean).to(device).float()
    
    return train_loader, valid_loader, test_loader, scaler, mean_scaler


def preprocess_wave_data(raw_file, output_path, dataset_type="NDBC1", station="42001"):
    """
    Preprocess raw NDBC buoy data.
    
    Args:
        raw_file: path to raw CSV/TXT file
        output_path: directory to save processed data
        dataset_type: "NDBC1" or "NDBC2"
        station: buoy station ID
    """
    import warnings
    warnings.filterwarnings('ignore')
    
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Read data
    # NDBC format: YYYY MM DD hh mm columns...
    df = pd.read_csv(raw_file, delim_whitespace=True)
    
    # Create datetime index
    df['datetime'] = pd.to_datetime(
        df[['#YY', 'MM', 'DD', 'hh', 'mm']].rename(
            columns={'#YY': 'year', 'MM': 'month', 'DD': 'day', 'hh': 'hour', 'mm': 'minute'}
        )
    )
    df.set_index('datetime', inplace=True)
    df.sort_index(inplace=True)
    
    # Select features (drop GST and WTMP based on paper)
    # Available: WDIR, WSPD, GST, WVHT, DPD, APD, MWD, PRES/BAR, ATMP, WTMP, DEWP
    feature_names = ['WDIR', 'WSPD', 'WVHT', 'DPD', 'APD', 'MWD', 'PRES', 'ATMP', 'DEWP']
    
    # Handle alternative column names
    if 'BAR' in df.columns and 'PRES' not in df.columns:
        df['PRES'] = df['BAR']
    
    # Select only available features
    available_features = [f for f in feature_names if f in df.columns]
    df = df[available_features]
    
    # Replace sentinel values with NaN
    # NDBC uses 99.0, 999, 999.0, etc. for missing data
    df = df.replace([99.0, 999, 999.0, 9999, 9999.0], np.nan)
    df = df.replace(99.00, np.nan)
    
    # Convert to numpy arrays
    data = df.values
    mask = (~np.isnan(data)).astype(float)
    
    # Compute mean/std from non-missing training data (exclude test portion)
    if dataset_type == "NDBC1":
        train_end = int(len(data) * 0.3)  # train+valid = 30%
    else:
        train_end = int(len(data) * 0.7)  # train = 70%
    
    train_data = data[:train_end]
    train_mean = np.nanmean(train_data, axis=0)
    train_std = np.nanstd(train_data, axis=0)
    
    # Replace NaN with 0 for masked positions (will be normalized)
    data = np.nan_to_num(data, nan=0.0)
    
    # Normalize
    normalized_data = (data - train_mean) / train_std
    normalized_data = normalized_data * mask  # Zero out missing positions
    
    # Save processed data
    processed_file = output_path / f"{dataset_type}_{station}_processed.pk"
    with open(processed_file, 'wb') as f:
        pickle.dump({
            'observed_data': normalized_data,
            'observed_mask': mask,
            'train_mean': train_mean,
            'train_std': train_std,
            'feature_names': available_features
        }, f)
    
    # Save mean/std separately
    meanstd_file = output_path / f"{dataset_type}_{station}_meanstd.pk"
    with open(meanstd_file, 'wb') as f:
        pickle.dump([train_mean, train_std], f)
    
    print(f"Processed data saved to {processed_file}")
    print(f"Data shape: {data.shape}")
    print(f"Missing ratio: {1 - mask.mean():.3f}")
    print(f"Features: {available_features}")
    
    return normalized_data, mask, train_mean, train_std
