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

    {'imu': ['lh', 'rh'], 'wifi': False, 'airpods': False, 'channel': 12, 'modality': 'imu', 'tag': '23'},
    {'imu': ['lp', 'rh'], 'wifi': False, 'airpods': False, 'channel': 12, 'modality': 'imu', 'tag': '24'},
    {'imu': ['lh', 'rp'], 'wifi': False, 'airpods': False, 'channel': 12, 'modality': 'imu', 'tag': '25'},
    {'imu': ['lh', 'lp'], 'wifi': False, 'airpods': False, 'channel': 12, 'modality': 'imu', 'tag': '26'},
    {'imu': ['gl', 'lh'], 'wifi': False, 'airpods': False, 'channel': 12, 'modality': 'imu', 'tag': '27'},
    {'imu': ['lp', 'rp'], 'wifi': False, 'airpods': False, 'channel': 12, 'modality': 'imu', 'tag': '28'},
    {'imu': ['gl', 'lp'], 'wifi': False, 'airpods': False, 'channel': 12, 'modality': 'imu', 'tag': '29'},
    {'imu': ['lh', 'rh', 'rp'], 'wifi': False, 'airpods': False, 'channel': 18, 'modality': 'imu', 'tag': '30'},
    {'imu': ['lh', 'lp', 'rh'], 'wifi': False, 'airpods': False, 'channel': 18, 'modality': 'imu', 'tag': '31'},
    {'imu': ['gl', 'lh', 'rh'], 'wifi': False, 'airpods': False, 'channel': 18, 'modality': 'imu', 'tag': '32'},
    {'imu': ['lp', 'rh', 'rp'], 'wifi': False, 'airpods': False, 'channel': 18, 'modality': 'imu', 'tag': '33'},
    {'imu': ['gl', 'lp', 'rh'], 'wifi': False, 'airpods': False, 'channel': 18, 'modality': 'imu', 'tag': '34'},
    {'imu': ['lh', 'lp', 'rp'], 'wifi': False, 'airpods': False, 'channel': 18, 'modality': 'imu', 'tag': '35'},
    {'imu': ['gl', 'lh', 'rp'], 'wifi': False, 'airpods': False, 'channel': 18, 'modality': 'imu', 'tag': '36'},
    {'imu': ['gl', 'lh', 'lp'], 'wifi': False, 'airpods': False, 'channel': 18, 'modality': 'imu', 'tag': '37'},
    {'imu': ['gl', 'lp', 'rp'], 'wifi': False, 'airpods': False, 'channel': 18, 'modality': 'imu', 'tag': '38'},
    {'imu': ['lh', 'lp', 'rh', 'rp'], 'wifi': False, 'airpods': False, 'channel': 24, 'modality': 'imu', 'tag': '39'},
    {'imu': ['gl', 'lh', 'rh', 'rp'], 'wifi': False, 'airpods': False, 'channel': 24, 'modality': 'imu', 'tag': '40'},
    {'imu': ['gl', 'lh', 'lp', 'rh'], 'wifi': False, 'airpods': False, 'channel': 24, 'modality': 'imu', 'tag': '41'},
    {'imu': ['gl', 'lp', 'rh', 'rp'], 'wifi': False, 'airpods': False, 'channel': 24, 'modality': 'imu', 'tag': '42'},
    {'imu': ['gl', 'lh', 'lp', 'rp'], 'wifi': False, 'airpods': False, 'channel': 24, 'modality': 'imu', 'tag': '43'},
    {'imu': ['gl', 'lh', 'lp', 'rh', 'rp'], 'wifi': False, 'airpods': False, 'channel': 30, 'modality': 'imu', 'tag': '44'},

    {'imu': ['rh'], 'wifi': False, 'airpods': True, 'channel': 12, 'modality': 'imu', 'tag': '45'},
    {'imu': ['lh'], 'wifi': False, 'airpods': True, 'channel': 12, 'modality': 'imu', 'tag': '46'},
    {'imu': ['lp'], 'wifi': False, 'airpods': True, 'channel': 12, 'modality': 'imu', 'tag': '47'},
    {'imu': ['gl'], 'wifi': False, 'airpods': True, 'channel': 12, 'modality': 'imu', 'tag': '48'},
    {'imu': ['lh', 'rh'], 'wifi': False, 'airpods': True, 'channel': 18, 'modality': 'imu', 'tag': '49'},
    {'imu': ['lp', 'rh'], 'wifi': False, 'airpods': True, 'channel': 18, 'modality': 'imu', 'tag': '50'},
    {'imu': ['gl', 'rh'], 'wifi': False, 'airpods': True, 'channel': 18, 'modality': 'imu', 'tag': '51'},
    {'imu': ['lh', 'rp'], 'wifi': False, 'airpods': True, 'channel': 18, 'modality': 'imu', 'tag': '52'},
    {'imu': ['lh', 'lp'], 'wifi': False, 'airpods': True, 'channel': 18, 'modality': 'imu', 'tag': '53'},
    {'imu': ['gl', 'lh'], 'wifi': False, 'airpods': True, 'channel': 18, 'modality': 'imu', 'tag': '54'},
    {'imu': ['lp', 'rp'], 'wifi': False, 'airpods': True, 'channel': 18, 'modality': 'imu', 'tag': '55'},
    {'imu': ['gl', 'lp'], 'wifi': False, 'airpods': True, 'channel': 18, 'modality': 'imu', 'tag': '56'},
    {'imu': ['lh', 'rh', 'rp'], 'wifi': False, 'airpods': True, 'channel': 24, 'modality': 'imu', 'tag': '57'},
    {'imu': ['lh', 'lp', 'rh'], 'wifi': False, 'airpods': True, 'channel': 24, 'modality': 'imu', 'tag': '58'},
    {'imu': ['gl', 'lh', 'rh'], 'wifi': False, 'airpods': True, 'channel': 24, 'modality': 'imu', 'tag': '59'},
    {'imu': ['lp', 'rh', 'rp'], 'wifi': False, 'airpods': True, 'channel': 24, 'modality': 'imu', 'tag': '60'},
    {'imu': ['gl', 'lp', 'rh'], 'wifi': False, 'airpods': True, 'channel': 24, 'modality': 'imu', 'tag': '61'},
    {'imu': ['lh', 'lp', 'rp'], 'wifi': False, 'airpods': True, 'channel': 24, 'modality': 'imu', 'tag': '62'},
    {'imu': ['gl', 'lh', 'rp'], 'wifi': False, 'airpods': True, 'channel': 24, 'modality': 'imu', 'tag': '63'},
    {'imu': ['gl', 'lh', 'lp'], 'wifi': False, 'airpods': True, 'channel': 24, 'modality': 'imu', 'tag': '64'},
    {'imu': ['gl', 'lp', 'rp'], 'wifi': False, 'airpods': True, 'channel': 24, 'modality': 'imu', 'tag': '65'},
    {'imu': ['lh', 'lp', 'rh', 'rp'], 'wifi': False, 'airpods': True, 'channel': 30, 'modality': 'imu', 'tag': '66'},
    {'imu': ['gl', 'lh', 'rh', 'rp'], 'wifi': False, 'airpods': True, 'channel': 30, 'modality': 'imu', 'tag': '67'},
    {'imu': ['gl', 'lh', 'lp', 'rh'], 'wifi': False, 'airpods': True, 'channel': 30, 'modality': 'imu', 'tag': '68'},
    {'imu': ['gl', 'lp', 'rh', 'rp'], 'wifi': False, 'airpods': True, 'channel': 30, 'modality': 'imu', 'tag': '69'},
    {'imu': ['gl', 'lh', 'lp', 'rp'], 'wifi': False, 'airpods': True, 'channel': 30, 'modality': 'imu', 'tag': '70'},
]

if __name__ == '__main__':
    day = get_day()
    args = init_configs()
    gpu = args.gpu
    run_list = {
        0: [
            ['device', receivers_to_keep_list[i], 'x', ('mamba', 8, 60, {'layer': 8, 'i': 1}), ('WWADLDatasetMutiAll', 'XRFV2', (30, 270), 'wifiimu')] for i in range(23, 27)
        ],
        1: [
            ['device', receivers_to_keep_list[i], 'x', ('mamba', 8, 60, {'layer': 8, 'i': 1}), ('WWADLDatasetMutiAll', 'XRFV2', (30, 270), 'wifiimu')] for i in range(27, 31)
        ],
        2: [
            ['device', receivers_to_keep_list[i], 'x', ('mamba', 8, 60, {'layer': 8, 'i': 1}), ('WWADLDatasetMutiAll', 'XRFV2', (30, 270), 'wifiimu')] for i in range(31, 35)
        ],
        3: [
            ['device', receivers_to_keep_list[i], 'x', ('mamba', 8, 60, {'layer': 8, 'i': 1}), ('WWADLDatasetMutiAll', 'XRFV2', (30, 270), 'wifiimu')] for i in range(35, 39)
        ],
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
