
from torch.utils.data import Dataset
from dataset.wwadl import WWADLDatasetSingle

class WWADLDataset(Dataset):
    def __init__(self, dataset_dir, split="train", modality_list = None):
        """
        初始化 WWADL 数据集。
        :param dataset_dir: 数据集所在目录路径。
        :param split: 数据集分割，"train" 或 "test"。
        """
        assert split in ["train", "test"], "split must be 'train' or 'test'"
     
