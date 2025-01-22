import sys
import os
project_path = '/root/shared-nvme/code/WWADL_code_mac'
dataset_root_path = '/root/shared-nvme/dataset'
causal_conv1d_path = '/root/shared-nvme/causal-conv1d'
mamba_path = '/root/shared-nvme/video-mamba-suite/mamba'
python_path = '/root/.conda/envs/t1/bin/python'
sys.path.append(project_path)
os.environ["PYTHONPATH"] = f"{project_path}:{causal_conv1d_path}:{mamba_path}:" + os.environ.get("PYTHONPATH", "")


def _to_var(data: dict, device):
    for key, value in data.items():
        data[key] = value.to(device)  # Directly move tensor to device
    return data


if __name__ == '__main__':

    cfg = {
        "model": {
            "name": "TAD_muti_none",
            # "backbone_name": "ActionMamba",
            "backbone_name": "Ushape",
            "modality": "imu",
            "in_channels": 30,
            # "embed_type": "Down",
            "backbone_config": None
        }
    }
    from dataset.wwadl import WWADLDatasetSingle, detection_collate
    from dataset.wwadl_muti import WWADLDatasetMuti
    from torch.utils.data import DataLoader
    # train_dataset = WWADLDatasetSingle('/root/shared-nvme/dataset/all_30_3', split='train', modality='imu')
    train_dataset = WWADLDatasetMuti('/root/shared-nvme/dataset/all_30_3')
    batch_size = 4
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=detection_collate,
        pin_memory=True,
        drop_last=True
    )

    from model.models import make_model, make_model_config
    model_cfg = make_model_config(cfg['model']['backbone_name'], cfg['model'])
    model = make_model(cfg['model']['name'], model_cfg).to('cuda')

    print(model.config.get_dict())

    for i, (data_batch, label_batch) in enumerate(train_data_loader):
        print(f"Batch {i} labels: {len(label_batch)}")
        data_batch = _to_var(data_batch, 'cuda')
        output = model(data_batch)

        break