import json
import os
import h5py
import torch

class WWADLDatasetTestSingle():

    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self.data_path = os.path.join(dataset_dir, f"test_data.h5")
        segment_label_path = os.path.join(dataset_dir, 'test_segment.json')

        self.info_path = os.path.join(dataset_dir, f"info.json")

        with open(self.info_path, 'r') as json_file:
            self.info = json.load(json_file)

        assert len(self.info['modality_list']) == 1, "single modality"

        self.modality = self.info['modality_list'][0]

        with open(segment_label_path, 'r') as json_file:
            self.segment_label = json.load(json_file)

        self.file_name_list = self.segment_label.keys()
        self.eval_gt = os.path.join(dataset_dir, f'{self.modality}_annotations.json')

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

                data = data.permute(1, 2, 0)
                data = data.reshape(-1, data.shape[-1])  # [5*6=30, 2048]

                yield data, segment

    def dataset(self):
        for file_name in self.file_name_list:
            yield file_name, self.get_data(file_name)

    def __len__(self):
        return len(self.file_name_list)