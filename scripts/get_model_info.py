import os
import sys
import json

# 定义路径
project_path = '/root/shared-nvme/code/WWADL_code_mac'
dataset_root_path = '/root/shared-nvme/dataset'
causal_conv1d_path = '/root/shared-nvme/video-mamba-suite/causal-conv1d'
mamba_path = '/root/shared-nvme/video-mamba-suite/mamba'
sys.path.append(project_path)

from model.models import make_model, make_model_config

os.environ["PYTHONPATH"] = f"{project_path}:{causal_conv1d_path}:{mamba_path}:" + os.environ.get("PYTHONPATH", "")

def load_setting(url: str)->dict:
    with open(url, 'r') as f:
        data = json.load(f)
        return data
    
test_model_list = [
    '/root/shared-nvme/code_result/result/25_01-20/muti_m_t/WWADLDatasetMuti_all_30_3_mamba_layer_8'
]

for test_model_path in test_model_list:
    config = load_setting(os.path.join(test_model_path, 'setting.json'))

    model_cfg = make_model_config(config['model']['backbone_name'], config['model'])
    model = make_model(config['model']['name'], model_cfg)

    log_info = 'model params: ' + str(sum(p.numel() for p in model.parameters() if p.requires_grad))
    print(log_info)

    log_info = 'model params(backbone): ' + str(sum(p.numel() for p in model.backbone.parameters() if p.requires_grad))
    print(log_info)
    
    if config['model']['name'] == 'TAD_muti_tsse':
        log_info = 'model params(embedding_tsse_imu): ' + str(sum(p.numel() for p in model.embedding_tsse_imu.parameters() if p.requires_grad))
        print(log_info)
        log_info = 'model params(embedding_tsse_wifi): ' + str(sum(p.numel() for p in model.embedding_tsse_wifi.parameters() if p.requires_grad))
        print(log_info)

    