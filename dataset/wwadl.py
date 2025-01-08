
import json
import os
import h5py
import torch
from torch.utils.data import Dataset


class WWADLDatasetSingle(Dataset):
    def __init__(self, dataset_dir, split="train"):
        """
        初始化 WWADL 数据集。
        :param dataset_dir: 数据集所在目录路径。
        :param split: 数据集分割，"train" 或 "test"。
        """
        assert split in ["train", "test"], "split must be 'train' or 'test'"
        self.dataset_dir = dataset_dir
        self.split = split

        self.data_path = os.path.join(dataset_dir, f"{split}_data.h5")
        self.label_path = os.path.join(dataset_dir, f"{split}_label.json")
        self.info_path = os.path.join(dataset_dir, f"info.json")

        with open(self.info_path, 'r') as json_file:
            self.info = json.load(json_file)

        assert len(self.info['modality_list']) == 1, "single modality"

        self.modality = self.info['modality_list'][0]

        with open(self.label_path, 'r') as json_file:
            self.labels = json.load(json_file)[self.modality]

        # self.h5_file = h5py.File(self.data_path, 'r')  # 打开文件
        # self.data = self.h5_file[self.modality]

    def shape(self):
        with h5py.File(self.data_path, 'r') as h5_file:
            data = h5_file[self.modality]
            shape = data.shape
        return shape

    def __len__(self):
        """
        返回数据集的样本数。
        """
        with h5py.File(self.data_path, 'r') as h5_file:
            data_length = h5_file[self.modality].shape[0]
        return data_length

    def __getitem__(self, idx):
        # 获取数据和标签
        # 延迟加载 h5 文件
        with h5py.File(self.data_path, 'r') as h5_file:
            sample = h5_file[self.modality][idx]

        label = self.labels[str(idx)]

        # 转换为 Tensor
        sample = torch.tensor(sample, dtype=torch.float32)

        if self.modality == 'imu':
            sample = sample.permute(1, 2, 0)  # [5, 6, 2048]
            sample = sample.reshape(-1, sample.shape[-1])  # [5*6=30, 2048]
        if self.modality == 'wifi':
            # [2048, 3, 3, 30]
            sample = sample.permute(1, 2, 3, 0)  # [3, 3, 30, 2048]
            sample = sample.reshape(-1, sample.shape[-1])  # [3*3*30, 2048]

        label = torch.tensor(label, dtype=torch.float32)

        # 将类别部分（最后一列）单独转换为整数
        label[:, -1] = label[:, -1].to(torch.long)  # 或者直接使用 label[:, -1] = label[:, -1].int()

        return sample, label

def detection_collate(batch):
    clips = []
    targets = []
    for sample in batch:
        clips.append(sample[0])
        target = sample[1]

        # 将类别部分单独转换为整数
        target[:, -1] = target[:, -1].to(torch.long)

        targets.append(target)
    return torch.stack(clips, 0), targets

if __name__ == '__main__':

    # train_dataset = WWADLDatasetSingle('/data/WWADL/dataset/imu_30_3', split='train')
    train_dataset = WWADLDatasetSingle('/root/shared-nvme/dataset/wifi_30_3', split='train')

    from torch.utils.data import DataLoader

    # 定义 DataLoader
    batch_size = 32
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        # worker_init_fn=worker_init_fn,
        collate_fn=detection_collate,
        pin_memory=True,
        drop_last=True
    )

    for i, (data_batch, label_batch) in enumerate(train_data_loader):
        print(f"Batch {i} data shape: {data_batch.shape}")
        print(f"Batch {i} labels: {len(label_batch)}")
        break

