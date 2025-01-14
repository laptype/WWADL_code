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
from dataset.modality.base import handle_nan_and_interpolate
from dataset.action import id_to_action


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

    def __init__(self, config, modality=None):
        # 初始化路径配置
        self.dataset_dir = config['path']['dataset_path']
        dataset_root_path = config['path']['dataset_root_path']
        self.test_file_list = load_file_list(self.dataset_dir)

        # 读取info.json文件
        self.info_path = os.path.join(self.dataset_dir, "info.json")
        with open(self.info_path, 'r') as json_file:
            self.info = json.load(json_file)

        if modality is None:
            assert len(self.info['modality_list']) == 1, "single modality"
            self.modality = self.info['modality_list'][0]
        else:
            self.modality = modality
        # 构建测试文件路径列表
        self.file_path_list = [
            os.path.join(dataset_root_path, self.modality, t)
            for t in self.test_file_list
        ]

        # 设置接收器过滤规则和新映射
        self.receivers_to_keep = self.info.get('receivers_to_keep', {
            "imu": None,
            "wifi": None,
            "airpods": None
        })
        self.new_mapping = self.info.get('new_mapping', None)

        # 定义模态数据集映射
        self.modality_dataset_map = {
            'imu': WWADL_imu,
            'wifi': WWADL_wifi,
            'airpods': WWADL_airpods
        }
        self.modality_dataset = self.modality_dataset_map[self.modality]

        # 加载分段和目标信息
        segment_info = self.info['segment_info']['train']
        self.clip_length = segment_info['window_len']
        self.stride = segment_info['window_step']
        self.target_len = self.info['segment_info']['target_len']

        # 加载评估标签路径
        self.eval_gt = os.path.join(self.dataset_dir, f'{self.modality}_annotations.json')

        # 加载全局均值和标准差
        self.global_mean, self.global_std = self.load_global_stats()

        # 初始化动作ID到动作映射
        self.id_to_action = self.info.get('id2action', id_to_action)

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

    def get_data(self, file_path):
        sample = self.modality_dataset(file_path,
                                       receivers_to_keep=self.receivers_to_keep[self.modality],
                                       new_mapping=self.new_mapping)
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

            # 调用封装的函数进行处理
            clip = handle_nan_and_interpolate(clip, self.clip_length, self.target_len)
            assert not np.any(np.isnan(clip)), "Data contains NaN values!"

            clip = torch.from_numpy(clip)  # 转为 Tensor

            if self.modality == 'imu':
                clip = clip.permute(1, 2, 0)  # [5, 6, 1500]
                clip = clip.reshape(-1, clip.shape[-1])  # [5*6=30, 1500]
            if self.modality == 'wifi':
                # [2048, 3, 3, 30]
                clip = clip.permute(1, 2, 3, 0)  # [3, 3, 30, 2048]
                clip = clip.reshape(-1, clip.shape[-1])  # [3*3*30, 2048]

            clip = clip.float()

            # 归一化
            clip = (clip - torch.tensor(self.global_mean, dtype=torch.float32)[:, None]) / \
                   (torch.tensor(self.global_std, dtype=torch.float32)[:, None] + 1e-6)

            yield clip, [offset, offset + self.clip_length]

    def dataset(self):
        for file_path, file_name in zip(self.file_path_list, self.test_file_list):
            yield file_name, self.get_data(file_path)

if __name__ == '__main__':
    dataset = WWADLDatasetTestSingle('/root/shared-nvme/WWADL', '/root/shared-nvme/dataset/imu_30_3')
    dataset.dataset()
