import sys
import os
import torch
import subprocess

# project_path = '/home/lanbo/WWADL/WWADL_code'
# dataset_root_path = '/data/WWADL/dataset'

# 定义路径
project_path = '/root/shared-nvme/code/WWADL_code_mac'
dataset_root_path = '/root/shared-nvme/dataset'
causal_conv1d_path = '/root/shared-nvme/causal-conv1d'
mamba_path = '/root/shared-nvme/video-mamba-suite/mamba'
python_path = '/root/.conda/envs/mamba/bin/python'
# python_path = '/root/.conda/envs/t1/bin/python'
sys.path.append(project_path)
os.environ["PYTHONPATH"] = f"{project_path}:{causal_conv1d_path}:{mamba_path}:" + os.environ.get("PYTHONPATH", "")

import argparse
from utils.setting import get_day, get_time, write_setting, get_result_path, get_log_path, Run_config, load_setting
from scripts.update_config import prepare_config
from global_config import get_basic_config

config = get_basic_config()

def init_configs():
    parser = argparse.ArgumentParser(description="WiVio study")
    parser.add_argument('--gpu', dest="gpu", required=True, type=int, default=0,
                        help="gpu")
    args = parser.parse_args()

    return args

receivers_to_keep_list = [
    {},
    {'imu': ['rh'], 'wifi': False, 'airpods': False, 'channel': 6, 'modality': 'imu', 'tag': '1'},
    {'imu': ['lh'], 'wifi': False, 'airpods': False, 'channel': 6, 'modality': 'imu', 'tag': '2'},
    {'imu': ['rp'], 'wifi': False, 'airpods': False, 'channel': 6, 'modality': 'imu', 'tag': '3'},
    {'imu': ['lp'], 'wifi': False, 'airpods': False, 'channel': 6, 'modality': 'imu', 'tag': '4'},
    {'imu': None, 'wifi': False, 'airpods': True, 'channel': 6, 'modality': 'imu', 'tag': '5'},
    {'imu': ['gl'], 'wifi': False, 'airpods': False, 'channel': 6, 'modality': 'imu', 'tag': '6'},
    {'imu': None, 'wifi': True, 'airpods': False, 'channel': 270, 'modality': 'wifi', 'tag': '7'},

    
    {'imu': ['rh'], 'wifi': True, 'airpods': False, 'channel': (6, 270), 'modality': 'wifimu', 'tag': '8'},
    {'imu': ['rh', 'rp'], 'wifi': False, 'airpods': False, 'channel': 12, 'modality': 'imu', 'tag': '9'},
    {'imu': ['rp'], 'wifi': False, 'airpods': True, 'channel': 12, 'modality': 'imu', 'tag': '10'},
    {'imu': ['rp', 'gl'], 'wifi': False, 'airpods': False, 'channel': 12, 'modality': 'imu', 'tag': '11'},
    {'imu': ['rh', 'gl'], 'wifi': False, 'airpods': False, 'channel': 12, 'modality': 'imu', 'tag': '12'},
    {'imu': ['rh', 'rp'], 'wifi': True, 'airpods': False, 'channel': (12, 270), 'modality': 'wifimu', 'tag': '13'},
    {'imu': ['rh', 'rp'], 'wifi': False, 'airpods': True, 'channel': 18, 'modality': 'imu', 'tag': '14'},

    {'imu': ['rh', 'rp', 'gl'], 'wifi': False, 'airpods': False, 'channel': 18, 'modality': 'imu', 'tag': '15'},
    {'imu': ['rp', 'gl'], 'wifi': False, 'airpods': True, 'channel': 18, 'modality': 'imu', 'tag': '16'},
    {'imu': ['rh', 'rp', 'gl'], 'wifi': True, 'airpods': False, 'channel': (18, 270), 'modality': 'wifimu', 'tag': '17'},
    {'imu': ['rh', 'rp', 'gl'], 'wifi': False, 'airpods': True, 'channel': 24, 'modality': 'imu', 'tag': '18'},
    {'imu': ['rh', 'rp', 'gl'], 'wifi': True, 'airpods': True, 'channel': (24, 270), 'modality': 'wifimu', 'tag': '19'},
    {'imu': ['rh', 'rp', 'lh', 'lp', 'gl'], 'wifi': False, 'airpods': True, 'channel': 36, 'modality': 'imu', 'tag': '20'},
    {'imu': ['rh', 'rp', 'lh', 'lp', 'gl'], 'wifi': True, 'airpods': True, 'channel': (36, 270), 'modality': 'imu', 'tag': '21'}, 
]


if __name__ == '__main__':
    day = get_day()
    args = init_configs()
    gpu = args.gpu
    run_list = {
        0: [
            ['device', receivers_to_keep_list[1], 'x', ('mamba', 8, 80, {'layer': 8, 'i': 1}), ('WWADLDatasetMutiAll', 'XRFV2', (30, 270), 'wifiimu')],
            ['device', receivers_to_keep_list[2], 'x', ('mamba', 8, 80, {'layer': 8, 'i': 1}), ('WWADLDatasetMutiAll', 'XRFV2', (30, 270), 'wifiimu')],
            ['device', receivers_to_keep_list[8], 'x', ('mamba', 8, 80, {'layer': 8, 'i': 1}), ('WWADLDatasetMutiAll', 'XRFV2', (30, 270), 'wifiimu')],
        ],
        1: [
            ['device', receivers_to_keep_list[9], 'x', ('mamba', 8, 80, {'layer': 8, 'i': 1}), ('WWADLDatasetMutiAll', 'XRFV2', (30, 270), 'wifiimu')],
            ['device', receivers_to_keep_list[12], 'x', ('mamba', 8, 80, {'layer': 8, 'i': 1}), ('WWADLDatasetMutiAll', 'XRFV2', (30, 270), 'wifiimu')],
            ['device', receivers_to_keep_list[13], 'x', ('mamba', 8, 80, {'layer': 8, 'i': 1}), ('WWADLDatasetMutiAll', 'XRFV2', (30, 270), 'wifiimu')],
        ],
        2: [
            ['device', receivers_to_keep_list[14], 'x', ('mamba', 8, 80, {'layer': 8, 'i': 1}), ('WWADLDatasetMutiAll', 'XRFV2', (30, 270), 'wifiimu')],
            ['device', receivers_to_keep_list[15], 'x', ('mamba', 8, 80, {'layer': 8, 'i': 1}), ('WWADLDatasetMutiAll', 'XRFV2', (30, 270), 'wifiimu')],
            ['device', receivers_to_keep_list[17], 'x', ('mamba', 8, 80, {'layer': 8, 'i': 1}), ('WWADLDatasetMutiAll', 'XRFV2', (30, 270), 'wifiimu')],
        ],
        3: [
            ['device', receivers_to_keep_list[18], 'x', ('mamba', 8, 80, {'layer': 8, 'i': 1}), ('WWADLDatasetMutiAll', 'XRFV2', (30, 270), 'wifiimu')],
            ['device', receivers_to_keep_list[19], 'x', ('mamba', 8, 80, {'layer': 8, 'i': 1}), ('WWADLDatasetMutiAll', 'XRFV2', (30, 270), 'wifiimu')],
            ['device', receivers_to_keep_list[20], 'x', ('mamba', 8, 80, {'layer': 8, 'i': 1}), ('WWADLDatasetMutiAll', 'XRFV2', (30, 270), 'wifiimu')],
        ]
    }

    for run in run_list[gpu]:
        tag, receivers_to_keep, model_arc_name, model_str, dataset_str, *_others = run
        dataset_name, dataset, channel, modality = dataset_str
        model_name, batch_size, epoch, model_config, *others = model_str
        tag = tag + receivers_to_keep['tag']

        if receivers_to_keep['modality'] == 'wifimu':
            model_arc_name = 'TAD_muti_weight_grc'
        else:
            model_arc_name = 'TAD_single'

        # 调用提取的函数
        updated_config = prepare_config(model_arc_name, dataset_str, model_str, config, gpu, day, dataset_root_path, python_path, tag, receivers_to_keep=receivers_to_keep)

        test_gpu = gpu

        # TRAIN =============================================================================================
        run = Run_config(config, 'train')

        train_command = (
            f"CUDA_VISIBLE_DEVICES={run.ddp_devices} {run.python_path} "
            f"{run.main_path} --is_train true --config_path {run.config_path}"
        )

        # 执行训练命令并等待其完成
        train_process = subprocess.run(train_command, shell=True)

        # 检查训练命令是否正常结束
        if train_process.returncode == 0:  # 正常结束返回 0
            config = load_setting(os.path.join(config['path']['result_path'], 'setting.json'))
            config['endtime'] = get_time()
            write_setting(config)

            # TEST ==========================================================================================
            test_command = (
                f"CUDA_VISIBLE_DEVICES={test_gpu} {run.python_path} "
                f"{run.main_path} --config_path {run.config_path}"
            )

            # 启动测试命令
            subprocess.run(test_command, shell=True)
        else:
            print("Training process failed. Test process will not start.")
