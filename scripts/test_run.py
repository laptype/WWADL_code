import os
import sys
import re
import json

# 定义路径
project_path = '/root/shared-nvme/code/WWADL_code_mac'
dataset_root_path = '/root/shared-nvme/dataset'
causal_conv1d_path = '/root/shared-nvme/video-mamba-suite/causal-conv1d'
mamba_path = '/root/shared-nvme/video-mamba-suite/mamba'
sys.path.append(project_path)

os.environ["PYTHONPATH"] = f"{project_path}:{causal_conv1d_path}:{mamba_path}:" + os.environ.get("PYTHONPATH", "")
import argparse
from utils.setting import get_day, get_time, write_setting, get_result_path, get_log_path, Run_config

def init_configs():
    parser = argparse.ArgumentParser(description="WiVio study")
    parser.add_argument('--gpu', dest="gpu", required=False, type=int, default=0,
                        help="gpu")
    args = parser.parse_args()

    return args

def load_setting(url: str)->dict:
    with open(url, 'r') as f:
        data = json.load(f)
        return data

def get_checkpoint_epoch_49(checkpoint_path):
    # 获取目录中的所有文件
    all_files = os.listdir(checkpoint_path)
    # print(all_files)
    
    # 正则表达式匹配 49 epoch 的文件名
    pattern = re.compile(r".*-epoch-(49)\.pt$")
    
    # 查找是否存在 epoch-49 的文件
    for file in all_files:
        if pattern.match(file):
            return file  # 找到就返回文件名
    
    return None  # 如果没有符合条件的文件

args = init_configs()
gpu = args.gpu

run_list = {
    0: [
        '/root/shared-nvme/code_result/result/25_01-30/persion2/WWADLDatasetMuti_all_2_30_3_mamba_layer_8_i_1',
        # '/root/shared-nvme/code_result/result/25_01-30/persion1/WWADLDatasetMuti_all_1_30_3_mamba_layer_8_i_1'
        # '/root/shared-nvme/code_result/result/25_01-26/persion3/WWADLDatasetMuti_all_3_30_3_mamba_layer_8_i_1'
        # '/root/shared-nvme/code_result/result/25_01-26/persion9/WWADLDatasetMuti_all_9_30_3_mamba_layer_8_i_1',
        # '/root/shared-nvme/code_result/result/25_01-26/persion14/WWADLDatasetMuti_all_9_30_3_mamba_layer_8_i_1'
        # '/root/shared-nvme/code_result/result/25_01-28/device1/WWADLDatasetMutiAll_XRFV2_mamba_layer_8_i_1-1',
        # '/root/shared-nvme/code_result/result/25_01-28/device2/WWADLDatasetMutiAll_XRFV2_mamba_layer_8_i_1-2',
        # '/root/shared-nvme/code_result/result/25_01-28/device8/WWADLDatasetMutiAll_XRFV2_mamba_layer_8_i_1-8',
        # '/root/shared-nvme/code_result/result/25_01-28/device9/WWADLDatasetMutiAll_XRFV2_mamba_layer_8_i_1-9', 
    ],
    1: [
        '/root/shared-nvme/code_result/result/25_01-30/persion4/WWADLDatasetMuti_all_4_30_3_mamba_layer_8_i_1'
        # '/root/shared-nvme/code_result/result/25_01-28/device12/WWADLDatasetMutiAll_XRFV2_mamba_layer_8_i_1-12',
        # '/root/shared-nvme/code_result/result/25_01-28/device13/WWADLDatasetMutiAll_XRFV2_mamba_layer_8_i_1-13',
        # '/root/shared-nvme/code_result/result/25_01-28/device14/WWADLDatasetMutiAll_XRFV2_mamba_layer_8_i_1-14',
        # '/root/shared-nvme/code_result/result/25_01-28/device15/WWADLDatasetMutiAll_XRFV2_mamba_layer_8_i_1-15',
    ],
    2: [
        '/root/shared-nvme/code_result/result/25_01-30/persion5/WWADLDatasetMuti_all_5_30_3_mamba_layer_8_i_1'
        # '/root/shared-nvme/code_result/result/25_01-28/device17/WWADLDatasetMutiAll_XRFV2_mamba_layer_8_i_1-17',
        # '/root/shared-nvme/code_result/result/25_01-28/device18/WWADLDatasetMutiAll_XRFV2_mamba_layer_8_i_1-18',
    ],
    3: [
        # '/root/shared-nvme/code_result/result/25_01-28/device19/WWADLDatasetMutiAll_XRFV2_mamba_layer_8_i_1-19',
        # '/root/shared-nvme/code_result/result/25_01-28/device20/WWADLDatasetMutiAll_XRFV2_mamba_layer_8_i_1-20',
    ]
}

test_model_list = run_list[gpu]

for test_model_path in test_model_list:
    config = load_setting(os.path.join(test_model_path, 'setting.json'))

    config['path']['dataset_root_path'] = '/root/shared-nvme/WWADL'
    # config['path']['dataset_path'] = '/root/shared-nvme/dataset/wifi_30_3'
    # config['dataset']['dataset_name'] = 'WWADLDatasetSingle'
    # config["model"]["modality"] = 'wifi'

    run = Run_config(config, 'train')

    test_gpu = gpu
    # config['testing']['pt_file_name'] = get_checkpoint_epoch_49(test_model_path)
    # config['model']['backbone_name'] = 'Transformer'

    # print(config['testing']['pt_file_name'])

    write_setting(config)

    print(run.config_path)

    run.python_path = '/root/.conda/envs/mamba/bin/python'

    os.system(
        f"CUDA_VISIBLE_DEVICES={test_gpu} {run.python_path} "
        f"{run.main_path} --config_path {run.config_path} "
    )