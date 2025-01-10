import os

class WWADLBase():
    def __init__(self, file_path):
        self.data = None
        self.label = None
        self.file_name = os.path.basename(file_path)


    def load_data(self, file_path):
        pass

    def show_info(self):
        print(self.data.shape)
        print(self.label)
