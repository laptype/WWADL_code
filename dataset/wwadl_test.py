import json
import os
import h5py
import torch
import numpy as np

class WWADLDatasetTestSingle():

    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self.data_path = os.path.join(dataset_dir, f"test_data.h5")
        segment_label_path = os.path.join(dataset_dir, 'test_segment.json')

        self.info_path = os.path.join(dataset_dir, f"info.json")
        self.stats_path = os.path.join(dataset_dir, "global_stats.json")  # 保存训练集均值和方差的路径

        with open(self.info_path, 'r') as json_file:
            self.info = json.load(json_file)

        assert len(self.info['modality_list']) == 1, "single modality"

        self.modality = self.info['modality_list'][0]

        with open(segment_label_path, 'r') as json_file:
            self.segment_label = json.load(json_file)

        self.file_name_list = self.segment_label.keys()
        self.eval_gt = os.path.join(dataset_dir, f'{self.modality}_annotations.json')

        # 加载全局均值和方差
        self.global_mean, self.global_std = self.load_global_stats()

    def load_global_stats(self):
        """
        从文件加载全局均值和方差。
        """
        stats_path = os.path.join(self.dataset_dir, "global_stats.json")
        if not os.path.exists(stats_path):
            raise FileNotFoundError(f"Global stats file '{stats_path}' not found. Ensure it is generated during training.")

        with open(stats_path, 'r') as f:
            stats = json.load(f)
        print(f"Loaded global stats from {stats_path}")
        return np.array(stats["global_mean"]), np.array(stats["global_std"])

    def get_data(self, file_name):
        with h5py.File(self.data_path, 'r') as h5_file:
            # Ensure that the file and modality exist in the HDF5 file
            if self.modality not in h5_file or file_name not in h5_file[self.modality]:
                raise ValueError(f"File '{file_name}' or modality '{self.modality}' not found in dataset.")

            # Iterate over all indices in the file
            for idx in h5_file[self.modality][file_name]:
                data = h5_file[self.modality][file_name][str(idx)][...]
                segment = self.segment_label[file_name][int(idx)]

                data = torch.tensor(data, dtype=torch.float32)

                # 全局归一化
                data = (data - torch.tensor(self.global_mean, dtype=torch.float32)) / \
                       (torch.tensor(self.global_std, dtype=torch.float32) + 1e-6)

                if self.modality == 'imu':
                    data = data.permute(1, 2, 0)
                    data = data.reshape(-1, data.shape[-1])  # [5*6=30, 2048]
                if self.modality == 'wifi':
                    # [2048, 3, 3, 30]
                    data = data.permute(1, 2, 3, 0)  # [3, 3, 30, 2048]
                    data = data.reshape(-1, data.shape[-1])  # [3*3*30, 2048]

                # 替换 NaN 和 Inf
                if torch.isnan(data).any() or torch.isinf(data).any():
                    raise ValueError("Input contains NaN or Inf values.")




                yield data, segment

    def dataset(self):
        for file_name in self.file_name_list:
            yield file_name, self.get_data(file_name)

    def __len__(self):
        return len(self.file_name_list)