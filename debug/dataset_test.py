import sys
import os
project_path = '/root/shared-nvme/code/WWADL_code_mac'
dataset_root_path = '/root/shared-nvme/dataset'
causal_conv1d_path = '/root/shared-nvme/causal-conv1d'
mamba_path = '/root/shared-nvme/video-mamba-suite/mamba'
python_path = '/root/.conda/envs/t1/bin/python'
sys.path.append(project_path)
os.environ["PYTHONPATH"] = f"{project_path}:{causal_conv1d_path}:{mamba_path}:" + os.environ.get("PYTHONPATH", "")


from dataset.wwadl_muti_all import WWADLDatasetMutiAll
from dataset.wwadl import WWADLDatasetSingle, detection_collate
from dataset.wwadl_muti_all_test import WWADLDatasetTestMutiALL
from dataset.wwadl_muti_test import WWADLDatasetTestMuti
if __name__ == '__main__':

    # from torch.utils.data import DataLoader

    receivers_to_keep = {
        'imu': ['lh', 'rh', 'lp', 'rp'],
        'wifi': True,
        'airpods': True
    }
    # receivers_to_keep = {
    #     'imu': None,
    #     'wifi': True,
    #     'airpods': None
    # }
    receivers_to_keep = None


    # dataset = WWADLDatasetMutiAll('/root/shared-nvme/dataset/XRFV2', split='train', receivers_to_keep=receivers_to_keep)
    # print(dataset.shape())

    # batch_size = 2
    # train_data_loader = DataLoader(
    #     dataset,
    #     batch_size=batch_size,
    #     shuffle=True,
    #     num_workers=1,
    #     collate_fn=detection_collate,
    #     pin_memory=True,
    #     drop_last=True
    # )

    # for i, (data_batch, label_batch) in enumerate(train_data_loader):
        
    #     for key, value in data_batch.items():
    #         print(f"Batch {i} data shape: {key} {value.shape}")

    #     break

    ###########################################################################################################

    from global_config import get_basic_config

    config = get_basic_config()

    config['path']['dataset_path'] = '/root/shared-nvme/dataset/XRFV2'

    dataset = WWADLDatasetTestMutiALL(config=config)

    # dataset = WWADLDatasetTestMuti(config=config)

    for file_name, data in dataset.dataset():
        print(file_name)
        for d, segment in data:
            for key, value in d.items():
                print(f"{key} {value.shape}", end=' ')
            print(segment)
            # break
        break