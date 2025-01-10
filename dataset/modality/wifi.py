
from dataset.modality.base import WWADLBase
from dataset.modality.h5 import load_h5

class WWADL_wifi(WWADLBase):
    def __init__(self, file_path):
        super().__init__(file_path)
        self.load_data(file_path)

    def load_data(self, file_path):

        data = load_h5(file_path)

        self.data = data['amp']
        self.label = data['label']














