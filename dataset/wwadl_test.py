import json
import os
import h5py
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

from dataset.modality.wifi import WWADL_wifi
from dataset.modality.imu import WWADL_imu
from dataset.modality.airpods import WWADL_airpods

from scipy.interpolate import interp1d

# class WWADLDataset():
#     def __init__(self, root_path, file_name_list, modality_list=None):
#         if modality_list is None:
#             modality_list = ['wifi', 'imu', 'airpods']
#         self.data_path = {
#             'wifi': os.path.join(root_path, 'wifi'),
#             'imu': os.path.join(root_path, 'imu'),
#             'airpods': os.path.join(root_path, 'AirPodsPro'),
#         }
#         # Initialize an empty dictionary for data
#         self.data = {}
#
#         # Only include modalities specified in the `modality` list
#         if 'wifi' in modality_list:
#             print("Loading WiFi data...")
#             self.data['wifi'] = [WWADL_wifi(os.path.join(self.data_path['wifi'], f)) for f in
#                                  tqdm(file_name_list, desc="WiFi files")]
#
#         if 'imu' in modality_list:
#             print("Loading IMU data...")
#             self.data['imu'] = [WWADL_imu(os.path.join(self.data_path['imu'], f)) for f in
#                                 tqdm(file_name_list, desc="IMU files")]
#
#         if 'airpods' in modality_list:
#             print("Loading AirPods data...")
#             self.data['airpods'] = [WWADL_airpods(os.path.join(self.data_path['airpods'], f)) for f in
#                                     tqdm(file_name_list, desc="AirPods files")]


def load_file_list(dataset_path):
    # 读取 test.csv
    test_csv_path = os.path.join(dataset_path, 'test.csv')
    if not os.path.exists(test_csv_path):
        raise FileNotFoundError(f"{test_csv_path} does not exist.")

    print("Loading test.csv...")
    test_df = pd.read_csv(test_csv_path)
    file_name_list = test_df['file_name'].tolist()
    print(f"Loaded {len(file_name_list)} file names from test.csv.")

    return file_name_list

class WWADLDatasetTestSingle():

    def __init__(self, config):
        dataset_dir = config['path']['dataset_path']
        dataset_root_path = config['path']['dataset_root_path']
        self.test_file_list = load_file_list(dataset_dir)

        self.info_path = os.path.join(dataset_dir, f"info.json")

        with open(self.info_path, 'r') as json_file:
            self.info = json.load(json_file)

        assert len(self.info['modality_list']) == 1, "single modality"
        self.modality = self.info['modality_list'][0]

        self.file_path_list = [os.path.join(dataset_root_path, self.modality, t) for t in self.test_file_list]

        modality_dataset = {
            'imu': WWADL_imu,
            'wifi': WWADL_wifi,
            'airpods': WWADL_airpods
        }

        self.clip_length = self.info['segment_info']['test']['window_len']
        self.stride = self.info['segment_info']['test']['window_step']
        self.target_len = self.info['segment_info']['target_len']

        self.modality_dataset = modality_dataset[self.modality]

        self.eval_gt = os.path.join(dataset_dir, f'{self.modality}_annotations.json')

    def get_data(self, file_path):
        sample = self.modality_dataset(file_path)
        sample_count = len(sample.data)

        # 生成 offset 列表，用于分割视频片段
        if sample_count < self.clip_length:
            offsetlist = [0]  # 视频长度不足 clip_length，只取一个片段
        else:
            offsetlist = list(range(0, sample_count - self.clip_length + 1, self.stride))  # 根据步长划分片段
            if (sample_count - self.clip_length) % self.stride:
                offsetlist += [sample_count - self.clip_length]  # 确保最后一个片段不被遗漏

        for offset in offsetlist:
            clip = sample.data[offset: offset + self.clip_length]  # 获取当前的 clip

            # 插值到 target_len 长度
            original_indices = np.linspace(0, self.clip_length - 1, self.clip_length)
            target_indices = np.linspace(0, self.clip_length - 1, self.target_len)
            clip = interp1d(original_indices, clip, axis=0, kind='linear')(target_indices)

            clip = torch.from_numpy(clip)  # 转为 Tensor

            clip = clip.permute(1, 2, 0)  # [5, 6, 1500]
            clip = clip.reshape(-1, clip.shape[-1])  # [5*6=30, 1500]

            clip = clip.float()

            yield clip, [offset, offset + self.clip_length]

    def dataset(self):
        for file_path, file_name in zip(self.file_path_list, self.test_file_list):
            yield file_name, self.get_data(file_path)




    #
    # def load_global_stats(self):
    #     """
    #     从文件加载全局均值和方差。
    #     """
    #     stats_path = os.path.join(self.dataset_dir, "global_stats.json")
    #     if not os.path.exists(stats_path):
    #         raise FileNotFoundError(f"Global stats file '{stats_path}' not found. Ensure it is generated during training.")
    #
    #     with open(stats_path, 'r') as f:
    #         stats = json.load(f)
    #     print(f"Loaded global stats from {stats_path}")
    #     return np.array(stats["global_mean"]), np.array(stats["global_std"])
    #
    # def get_data(self, file_name):
    #     with h5py.File(self.data_path, 'r') as h5_file:
    #         # Ensure that the file and modality exist in the HDF5 file
    #         if self.modality not in h5_file or file_name not in h5_file[self.modality]:
    #             raise ValueError(f"File '{file_name}' or modality '{self.modality}' not found in dataset.")
    #
    #         # Iterate over all indices in the file
    #         for idx in h5_file[self.modality][file_name]:
    #             data = h5_file[self.modality][file_name][str(idx)][...]
    #             segment = self.segment_label[file_name][int(idx)]
    #
    #             data = torch.tensor(data, dtype=torch.float32)
    #
    #             if self.modality == 'imu':
    #                 data = data.permute(1, 2, 0)
    #                 data = data.reshape(-1, data.shape[-1])  # [5*6=30, 2048]
    #             if self.modality == 'wifi':
    #                 # [2048, 3, 3, 30]
    #                 data = data.permute(1, 2, 3, 0)  # [3, 3, 30, 2048]
    #                 data = data.reshape(-1, data.shape[-1])  # [3*3*30, 2048]
    #
    #             # 替换 NaN 和 Inf
    #             if torch.isnan(data).any() or torch.isinf(data).any():
    #                 raise ValueError("Input contains NaN or Inf values.")
    #
    #             # 全局归一化
    #             # data = (data - torch.tensor(self.global_mean, dtype=torch.float32)[:, None]) / \
    #             #        (torch.tensor(self.global_std, dtype=torch.float32)[:, None] + 1e-6)
    #
    #             yield data, segment
    #
    # def dataset(self):
    #     for file_name in self.file_name_list:
    #         yield file_name, self.get_data(file_name)
    #
    # def __len__(self):
    #     return len(self.file_name_list)

if __name__ == '__main__':
    dataset = WWADLDatasetTestSingle('/root/shared-nvme/WWADL', '/root/shared-nvme/dataset/imu_30_3')
    dataset.dataset()
