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

if __name__ == '__main__':
    dataset = WWADLDatasetTestSingle('/root/shared-nvme/WWADL', '/root/shared-nvme/dataset/imu_30_3')
    dataset.dataset()
