from dataset.modality.base import WWADLBase
from utils.h5 import load_h5


class WWADL_airpods(WWADLBase):
    def __init__(self, file_path, receivers_to_keep = None, new_mapping=None):
        super().__init__(file_path)
        self.duration = 0
        self.load_data(file_path)
        if new_mapping:
            self.mapping_label(new_mapping)


    def load_data(self, file_path):
        data = load_h5(file_path)

        self.data = data['data']
        self.label = data['label']
        self.duration = data['duration']



