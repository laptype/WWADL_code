import os
import json
from utils.setting import get_day, get_time, write_setting, get_result_path, get_log_path, Run_config

def load_setting(url: str)->dict:
    with open(url, 'r') as f:
        data = json.load(f)
        return data

test_model_list = [
    # '/root/shared-nvme/code_result/result/25_01-09/model_size/WWADLDatasetSingle_wifi_30_3_34_2048_270_1',
    # '/root/shared-nvme/code_result/result/25_01-09/model_size/WWADLDatasetSingle_imu_30_3_34_2048_30_1',
    # '/root/shared-nvme/code_result/result/25_01-09/device/WWADLDatasetSingle_wifi_30_3_0_34_2048_90_0',
    '/root/shared-nvme/code_result/result/25_01-10/test/WWADLDatasetSingle_imu_30_3_34_2048_30_0'
]


for test_model_path in test_model_list:
    config = load_setting(os.path.join(test_model_path, 'setting.json'))

    run = Run_config(config, 'train')

    test_gpu = 0

    print(run.config_path)

    os.system(
        f"CUDA_VISIBLE_DEVICES={test_gpu} {run.python_path} "
        f"{run.main_path} --config_path {run.config_path} "
    )