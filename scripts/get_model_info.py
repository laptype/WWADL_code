import os
import sys
import json
import torch
from fvcore.nn import FlopCountAnalysis


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
    '/root/shared-nvme/code_result/result/25_01-23/fusion_grc/WWADLDatasetMuti_all_30_3_mamba_layer_8_i_1',
    # '/root/shared-nvme/code_result/result/25_01-21/muti_w/WWADLDatasetMuti_all_30_3_wifiTAD'
    # '/root/shared-nvme/code_result/result/25_01-22/TriDet/WWADLDatasetMuti_all_30_3_TriDet_i_2',
    # '/root/shared-nvme/code_result/result/25_01-22/TemporalMaxer/WWADLDatasetMuti_all_30_3_TemporalMaxer_i_1',
    # '/root/shared-nvme/code_result/result/25_01-22/ActionMamba/WWADLDatasetMuti_all_30_3_ActionMamba_layer_8_i_1'
    # '/root/shared-nvme/code_result/result/25_01-22/ActionFormer/WWADLDatasetMuti_all_30_3_ActionFormer_layer_8_i_1',
    # '/root/shared-nvme/code_result/result/25_01-20/muti_m_t/WWADLDatasetMuti_all_30_3_mamba_layer_8',
    # '/root/shared-nvme/code_result/result/25_01-23/ushape/WWADLDatasetMuti_all_30_3_Ushape_layer_8_i_1'
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
    

    model = model.to('cuda')

    # 假设输入的形状为 B=32, imu_channels=30, wifi_channels=270, seq_length=2048
    B = 4  # 批大小
    imu_data = torch.randn(B, 30, 2048).to('cuda')  # 将 imu 数据移动到 GPU
    wifi_data = torch.randn(B, 270, 2048).to('cuda')  # 将 wifi 数据移动到 GPU
    # 创建输入字典
    inputs = {'imu': imu_data, 'wifi': wifi_data}
    # 计算 FLOPS
    flops = FlopCountAnalysis(model, inputs)
    # 计算每个样本的 FLOPS（总 FLOPS 除以批次大小）
    flops_per_sample = flops.total() / B

    # 转换为 GFLOPS（除以 1,000,000,000）
    gflops_per_sample = flops_per_sample / 1e9

    print(f'Total GFLOPS per sample: {gflops_per_sample}')
    