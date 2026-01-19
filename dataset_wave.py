import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from pathlib import Path


class Wave_Dataset(Dataset):
    def __init__(self, dataset_type="NDBC1", mode="train", validindex=0,
                 station="42001", data_path="./data/wave"):
        self.dataset_type = dataset_type
        self.mode = mode
        self.validindex = validindex

        if "," in station:
            self.stations = [s.strip() for s in station.split(",")]
        else:
            self.stations = [station]

        self.data_path = Path(data_path)

        if dataset_type == "NDBC1":
            self.eval_length = 72
        elif dataset_type == "NDBC2":
            self.eval_length = 36
        else:
            raise ValueError(f"Unknown dataset_type: {dataset_type}")

        self.feature_names = ['WDIR', 'WSPD', 'WVHT', 'DPD', 'APD', 'MWD', 'PRES', 'ATMP', 'DEWP']
        self.target_dim = len(self.feature_names)

        self._load_data()
        self._create_splits()

    def _load_data(self):
        all_obs = []
        all_masks = []

        first_station = self.stations[0]
        meanstd_path = self.data_path / f"{self.dataset_type}_{first_station}_meanstd.pk"

        if meanstd_path.exists():
            with open(meanstd_path, "rb") as f:
                self.train_mean, self.train_std = pickle.load(f)
        else:
            self.train_mean = None
            self.train_std = None

        print(f"Loading stations: {self.stations}")

        for st in self.stations:
            data_file = self.data_path / f"{self.dataset_type}_{st}_processed.pk"

            if not data_file.exists():
                raise FileNotFoundError(
                    f"Processed data file not found: {data_file}\nPlease run preprocessing for station {st} first."
                )

            with open(data_file, "rb") as f:
                data_dict = pickle.load(f)
                all_obs.append(data_dict['observed_data'])
                all_masks.append(data_dict['observed_mask'])

                if self.train_mean is None:
                    self.train_mean = data_dict['train_mean']
                    self.train_std = data_dict['train_std']

        self.observed_data = np.concatenate(all_obs, axis=0)
        self.observed_mask = np.concatenate(all_masks, axis=0)

        print(f"Combined data shape: {self.observed_data.shape}")

    def _create_splits(self):
        total_length = len(self.observed_data)

        if self.dataset_type == "NDBC1":
            train_ratio = 0.2
            valid_ratio = 0.1

            train_end = int(total_length * train_ratio)
            valid_end = train_end + int(total_length * valid_ratio)

            if self.mode == "train":
                self.use_index = np.arange(0, train_end - self.eval_length + 1, 1)
            elif self.mode == "valid":
                self.use_index = np.arange(train_end, valid_end - self.eval_length + 1, 1)
            elif self.mode == "test":
                self.use_index = np.arange(valid_end, total_length - self.eval_length + 1, self.eval_length)

        elif self.dataset_type == "NDBC2":
            train_ratio = 0.7
            valid_ratio = 0.1

            train_end = int(total_length * train_ratio)
            valid_end = train_end + int(total_length * valid_ratio)

            if self.mode == "train":
                self.use_index = np.arange(0, train_end - self.eval_length + 1, 1)
            elif self.mode == "valid":
                self.use_index = np.arange(train_end, valid_end - self.eval_length + 1, 1)
            elif self.mode == "test":
                self.use_index = np.arange(valid_end, total_length - self.eval_length + 1, self.eval_length)

        self.cut_length = [0] * len(self.use_index)

        if self.mode == "test" and len(self.observed_data) % self.eval_length != 0:
            remainder = len(self.observed_data) % self.eval_length
            if remainder > 0 and self.use_index[-1] + self.eval_length > len(self.observed_data):
                self.cut_length[-1] = self.eval_length - remainder

    def __getitem__(self, org_index):
        index = self.use_index[org_index]
        obs_data = self.observed_data[index:index + self.eval_length]
        obs_mask = self.observed_mask[index:index + self.eval_length]

        gt_mask = obs_mask.copy()

        if self.mode != "train":
            if self.dataset_type == "NDBC1":
                for t in range(0, self.eval_length, 18):
                    if t + 6 <= self.eval_length:
                        gt_mask[t:t + 6, 2] = 0
            elif self.dataset_type == "NDBC2":
                observed_indices = np.where(obs_mask > 0)
                if len(observed_indices[0]) > 0:
                    n_observed = len(observed_indices[0])
                    n_to_mask = int(n_observed * 0.1)
                    if n_to_mask > 0:
                        mask_indices = np.random.choice(n_observed, n_to_mask, replace=False)
                        gt_mask[
                            observed_indices[0][mask_indices],
                            observed_indices[1][mask_indices]
                        ] = 0

            gt_mask = gt_mask * obs_mask

        s = {
            'observed_data': obs_data,
            'observed_mask': obs_mask,
            'gt_mask': gt_mask,
            'timepoints': np.arange(self.eval_length) * 1.0,
            'cut_length': self.cut_length[org_index],
            'hist_mask': obs_mask.copy()
        }
        return s

    def __len__(self):
        return len(self.use_index)


class Wave_Dataset_Forecasting(Dataset):
    def __init__(self, dataset_type="NDBC1", mode="train",
                 station="42001", data_path="./data/wave",
                 history_length=None, pred_length=None):
        self.dataset_type = dataset_type
        self.mode = mode
        self.station = station
        self.data_path = Path(data_path)

        if dataset_type == "NDBC1":
            self.history_length = 30 if history_length is None else history_length
            self.pred_length = 6 if pred_length is None else pred_length
        elif dataset_type == "NDBC2":
            self.history_length = 168 if history_length is None else history_length
            self.pred_length = 24 if pred_length is None else pred_length

        self.seq_length = self.history_length + self.pred_length

        self._load_data()
        self._create_splits()

    def _load_data(self):
        imputed_file = self.data_path / f"{self.dataset_type}_{self.station}_imputed.pk"
        processed_file = self.data_path / f"{self.dataset_type}_{self.station}_processed.pk"

        if imputed_file.exists():
            with open(imputed_file, "rb") as f:
                data_dict = pickle.load(f)
                self.main_data = data_dict['imputed_data']
                self.mask_data = np.ones_like(self.main_data)
                self.mean_data = data_dict['train_mean']
                self.std_data = data_dict['train_std']
        elif processed_file.exists():
            with open(processed_file, "rb") as f:
                data_dict = pickle.load(f)
                obs_data = data_dict['observed_data']
                obs_mask = data_dict['observed_mask']
                self.main_data = obs_data.copy()
                for i in range(obs_data.shape[1]):
                    mean_val = np.nanmean(obs_data[:, i])
                    self.main_data[obs_mask[:, i] == 0, i] = mean_val
                self.mask_data = obs_mask
                self.mean_data = data_dict['train_mean']
                self.std_data = data_dict['train_std']
        else:
            raise FileNotFoundError(f"No data found at {self.data_path}")

        self.main_data = (self.main_data - self.mean_data) / self.std_data

    def _create_splits(self):
        total_length = len(self.main_data)

        if self.mode == "train":
            self.use_index = np.arange(
                0, total_length - self.seq_length - 320 * self.pred_length + 1, 1
            )
        else:
            self.use_index = np.arange(
                total_length - 320 * self.pred_length,
                total_length - self.seq_length + 1,
                self.pred_length
            )

    def __getitem__(self, org_index):
        index = self.use_index[org_index]
        data = self.main_data[index:index + self.seq_length]
        mask = self.mask_data[index:index + self.seq_length]

        target_mask = mask.copy()
        target_mask[-self.pred_length:] = 0.0

        return {
            'observed_data': data,
            'observed_mask': mask,
            'gt_mask': target_mask,
            'timepoints': np.arange(self.seq_length) * 1.0,
            'feature_id': np.arange(data.shape[1]) * 1.0,
        }

    def __len__(self):
        return len(self.use_index)
