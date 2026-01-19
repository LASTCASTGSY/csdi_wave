import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from pathlib import Path


class Wave_Dataset(Dataset):
    """
    Dataset class for NDBC buoy wave data (Multi-Station Support).
    Supports two dataset types:
    - NDBC1: 10-min interval data (2017-2018) with 9 features, eval_length=72
    - NDBC2: 1-hour interval data (2016) with 9 features, eval_length=36
    """
    def __init__(self, dataset_type="NDBC1", mode="train", validindex=0, 
                 station="42001", data_path="./data/wave"):
        self.dataset_type = dataset_type
        self.mode = mode
        self.validindex = validindex
        
        # --- FIX 1: Handle Multiple Stations ---
        # Split comma-separated string into a list
        if "," in station:
            self.stations = [s.strip() for s in station.split(",")]
        else:
            self.stations = [station]
            
        self.data_path = Path(data_path)
        
        # Set eval_length based on dataset type
        if dataset_type == "NDBC1":
            self.eval_length = 72
        elif dataset_type == "NDBC2":
            self.eval_length = 36
        else:
            raise ValueError(f"Unknown dataset_type: {dataset_type}")
        
        self.feature_names = ['WDIR', 'WSPD', 'WVHT', 'DPD', 'APD', 'MWD', 'PRES', 'ATMP', 'DEWP']
        self.target_dim = len(self.feature_names)
        
        # Load and concatenate data
        self._load_data()
        
        # Create train/val/test splits
        self._create_splits()
    
    def _load_data(self):
        """Load and concat multiple stations"""
        all_obs = []
        all_masks = []
        
        # --- FIX 2: Load Mean/Std from the FIRST station ---
        # We need these stats to un-normalize later. We use the first station's stats as reference.
        first_station = self.stations[0]
        meanstd_path = self.data_path / f"{self.dataset_type}_{first_station}_meanstd.pk"
        
        if meanstd_path.exists():
            with open(meanstd_path, "rb") as f:
                self.train_mean, self.train_std = pickle.load(f)
        else:
            # Fallback or error if missing
            self.train_mean = None
            self.train_std = None

        print(f"Loading stations: {self.stations}")

        # --- FIX 3: Loop through stations and accumulate data ---
        for st in self.stations:
            data_file = self.data_path / f"{self.dataset_type}_{st}_processed.pk"
            
            if not data_file.exists():
                 raise FileNotFoundError(f"Processed data file not found: {data_file}\nPlease run preprocessing for station {st} first.")
            
            with open(data_file, "rb") as f:
                data_dict = pickle.load(f)
                all_obs.append(data_dict['observed_data'])
                all_masks.append(data_dict['observed_mask'])
                
                # If mean/std wasn't loaded from file (legacy check), grab it here
                if self.train_mean is None:
                    self.train_mean = data_dict['train_mean']
                    self.train_std = data_dict['train_std']

        # --- FIX 4: Concatenate all stations along the time axis ---
        # This creates one giant timeline containing all data from all buoys
        self.observed_data = np.concatenate(all_obs, axis=0)
        self.observed_mask = np.concatenate(all_masks, axis=0)
        
        print(f"Combined data shape: {self.observed_data.shape}")
    
    # ... (Keep _create_splits, __getitem__, and __len__ exactly as they were) ...
    def _create_splits(self):
        """Create train/valid/test indices"""
        total_length = len(self.observed_data)
        
        if self.dataset_type == "NDBC1":
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
                self.use_index = np.arange(start, end, self.eval_length)
        
        elif self.dataset_type == "NDBC2":
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
        
        self.cut_length = [0] * len(self.use_index)
        if self.mode == "test" and len(self.observed_data) % self.eval_length != 0:
            remainder = len(self.observed_data) % self.eval_length
            if remainder > 0 and self.use_index[-1] + self.eval_length > len(self.observed_data):
                self.cut_length[-1] = self.eval_length - remainder
    
    def __getitem__(self, org_index):
        index = self.use_index[org_index]
        obs_data = self.observed_data[index:index + self.eval_length]
        obs_mask = self.observed_mask[index:index + self.eval_length]
        
        if self.mode == "train":
            gt_mask = obs_mask.copy()
        else:
            gt_mask = obs_mask.copy()
            if self.dataset_type == "NDBC1":
                WVHT_INDEX = 2
                for t in range(0, self.eval_length, 18):
                    if t + 6 <= self.eval_length:
                        gt_mask[t:t+6, WVHT_INDEX] = 0
            elif self.dataset_type == "NDBC2":
                observed_indices = np.where(obs_mask > 0)
                if len(observed_indices[0]) > 0:
                    n_observed = len(observed_indices[0])
                    n_to_mask = int(n_observed * 0.1)
                    if n_to_mask > 0:
                        mask_indices = np.random.choice(n_observed, n_to_mask, replace=False)
                        gt_mask[observed_indices[0][mask_indices], observed_indices[1][mask_indices]] = 0
            else:
                gt_mask = obs_mask.copy()
            gt_mask = gt_mask * obs_mask

        timepoints = np.arange(self.eval_length) * 1.0
        s = {
            'observed_data': obs_data,
            'observed_mask': obs_mask,
            'gt_mask': gt_mask,
            'timepoints': timepoints,
            'cut_length': self.cut_length[org_index],
            'hist_mask': obs_mask.copy()
        }
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
    # Note: Column names vary by year. Older files use #YY, newer use YY
    df = pd.read_csv(raw_file, delim_whitespace=True)
    # Convert time columns to numeric
    for col in ['#YY', 'MM', 'DD', 'hh', 'mm', 'YY', '#yr', 'yr', 'mo', 'dy', 'hr', 'mn']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    
    # Print actual columns for debugging
    print(f"\nDetected columns: {list(df.columns[:10])}...")
    
    # Handle different NDBC formats
    # Try to identify time columns (they vary by year)
    time_cols = []
    if '#YY' in df.columns:
        time_cols = ['#YY', 'MM', 'DD', 'hh', 'mm']
        year_col = '#YY'
    elif 'YY' in df.columns:
        time_cols = ['YY', 'MM', 'DD', 'hh', 'mm']
        year_col = 'YY'
    elif '#yr' in df.columns:
        time_cols = ['#yr', 'mo', 'dy', 'hr', 'mn']
        year_col = '#yr'
    elif 'yr' in df.columns:
        time_cols = ['yr', 'mo', 'dy', 'hr', 'mn']
        year_col = 'yr'
    else:
        raise ValueError(f"Cannot identify time columns. Available columns: {list(df.columns)}")
    
    print(f"Using time columns: {time_cols}")
    
    # Create datetime index
    # Handle 2-digit vs 4-digit years
    year_values = df[year_col].values
    if year_values.max() < 100:
        # 2-digit year: convert to 4-digit (assume 1900s for <50, 2000s for >=50)
        year_values = np.where(year_values < 50, year_values + 2000, year_values + 1900)
    
    df['datetime'] = pd.to_datetime({
        'year': year_values,
        'month': df[time_cols[1]],
        'day': df[time_cols[2]],
        'hour': df[time_cols[3]],
        'minute': df[time_cols[4]]
    })
    df.set_index('datetime', inplace=True)
    df.sort_index(inplace=True)
    
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    print(f"Total records: {len(df)}")
    
    # Select features (drop GST and WTMP based on paper)
    # Available: WDIR, WSPD, GST, WVHT, DPD, APD, MWD, PRES/BAR, ATMP, WTMP, DEWP
    # Note: Column names may vary (e.g., PRES vs BAR, ATMP vs ATMP)
    feature_names = ['WDIR', 'WSPD', 'WVHT', 'DPD', 'APD', 'MWD', 'PRES', 'ATMP', 'DEWP']
    
    # Handle alternative column names
    column_mapping = {
        'BAR': 'PRES',      # Older files use BAR instead of PRES
        'WTMP': 'WTMP',     # Keep for now, will drop later
        'GST': 'GST',       # Keep for now, will drop later
    }
    
    # Apply column renaming if needed
    for old_name, new_name in column_mapping.items():
        if old_name in df.columns and new_name not in df.columns:
            df.rename(columns={old_name: new_name}, inplace=True)
    
    # Select only available features
    available_features = [f for f in feature_names if f in df.columns]
    
    if len(available_features) == 0:
        raise ValueError(f"No expected features found! Available columns: {list(df.columns)}")
    
    if 'WVHT' not in available_features:
        raise ValueError("WVHT (wave height) column not found! Cannot proceed.")
    
    print(f"\nSelected features ({len(available_features)}): {available_features}")
    
    df = df[available_features]

# Force all selected features to numeric (strings → NaN)
    for col in available_features:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    
    # CRITICAL: Replace ALL sentinel missing values with NaN
    # NDBC uses different codes for different variables:
    # - WVHT, DPD, APD, WSPD, ATMP, WTMP, DEWP: 99.00 or 99 or 99.0
    # - WDIR, MWD: 999 
    # - PRES/BAR: 9999 or 9999.0
    
    # First pass: replace exact matches
    df = df.replace({
        99: np.nan,
        99.0: np.nan, 
        99.00: np.nan,
        999: np.nan,
        999.0: np.nan,
        9999: np.nan,
        9999.0: np.nan,
    })
    
    # Second pass: use threshold-based replacement for safety
    # Any value >= 99 in wave/wind/temp columns is likely a missing code
    for col in available_features:
        if col in ['WVHT', 'DPD', 'APD', 'WSPD', 'GST', 'ATMP', 'WTMP', 'DEWP']:
            # These should be < 50 normally, so >= 99 is definitely missing
            df.loc[df[col] >= 99, col] = np.nan
        elif col in ['WDIR', 'MWD']:
            # Direction: valid is 0-360, so > 360 is missing
            df.loc[df[col] > 360, col] = np.nan
        elif col in ['PRES', 'BAR']:
            # Pressure: valid is ~900-1100 hPa, so > 1100 or < 900 is suspect
            df.loc[(df[col] > 1100) | (df[col] < 900), col] = np.nan
    
    # Third pass: Replace any remaining unlikely values
    # If a value is exactly 99, 999, 9999 (common missing codes), remove it
    df = df.replace([99, 99.0, 99.00, 999, 999.0, 9999, 9999.0], np.nan)
    
    print(f"Missing value statistics after cleaning:")
    print(df.isnull().sum())
    print(f"Total missing ratio: {df.isnull().sum().sum() / (len(df) * len(df.columns)):.1%}")
    
    # Convert to numpy arrays
    data = df.values
    mask = (~np.isnan(data)).astype(float)
    
    # CRITICAL SANITY CHECK: Verify WVHT is clean
    wvht_idx = available_features.index('WVHT') if 'WVHT' in available_features else -1
    if wvht_idx >= 0:
        wvht_values = data[:, wvht_idx]
        wvht_observed = wvht_values[~np.isnan(wvht_values)]
        print(f"\nWVHT (Significant Wave Height) statistics:")
        print(f"  Min: {np.min(wvht_observed):.3f} m")
        print(f"  Max: {np.max(wvht_observed):.3f} m")
        print(f"  Mean: {np.mean(wvht_observed):.3f} m")
        print(f"  Std: {np.std(wvht_observed):.3f} m")
        print(f"  Missing: {np.isnan(wvht_values).sum()} / {len(wvht_values)} ({np.isnan(wvht_values).mean():.1%})")
        
        # Final sanity check: WVHT should be 0-10 m normally
        if np.max(wvht_observed) > 15:
            print(f"  ⚠️  WARNING: Maximum WVHT is {np.max(wvht_observed):.1f}m - checking for missing codes...")
            # Find outliers
            outliers = wvht_observed[wvht_observed > 15]
            print(f"  ⚠️  Found {len(outliers)} values > 15m: {np.unique(outliers)}")
            print(f"  ⚠️  These are likely missing codes! Removing them...")
            # Remove outliers
            data[wvht_values > 15, wvht_idx] = np.nan
            mask[wvht_values > 15, wvht_idx] = 0
    
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
