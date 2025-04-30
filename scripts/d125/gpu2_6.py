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
# python_path = '/root/.conda/envs/mamba/bin/python'
python_path = '/root/.conda/envs/t1/bin/python'
sys.path.append(project_path)
os.environ["PYTHONPATH"] = f"{project_path}:{causal_conv1d_path}:{mamba_path}:" + os.environ.get("PYTHONPATH", "")


from utils.setting import get_day, get_time, write_setting, get_result_path, get_log_path, Run_config, load_setting
from scripts.update_config import prepare_config
from global_config import get_basic_config

config = get_basic_config()


if __name__ == '__main__':

    receivers_to_keep_list = [
        # {'imu': ['rh'], 'wifi': False, 'airpods': False, 'channel': 6, 'modality': 'imu', 'tag': 'rh'},
        # {'imu': ['lh'], 'wifi': False, 'airpods': False, 'channel': 6, 'modality': 'imu', 'tag': 'lh'},
        # {'imu': ['rp'], 'wifi': False, 'airpods': False, 'channel': 6, 'modality': 'imu', 'tag': 'rp'},
        # {'imu': ['lp'], 'wifi': False, 'airpods': False, 'channel': 6, 'modality': 'imu', 'tag': 'lp'},
        # {'imu': None, 'wifi': False, 'airpods': True, 'channel': 6, 'modality': 'imu', 'tag': 'ar'},
        # {'imu': ['gl'], 'wifi': False, 'airpods': False, 'channel': 6, 'modality': 'imu', 'tag': 'gl'},
        # {'imu': None, 'wifi': True, 'airpods': False, 'channel': 270, 'modality': 'wifi', 'tag': 'wi'},

        # {'imu': ['rh', 'rp'], 'wifi': False, 'airpods': False, 'channel': 12, 'modality': 'imu', 'tag': 'rhrp'},
        # {'imu': ['rp'], 'wifi': False, 'airpods': True, 'channel': 12, 'modality': 'imu', 'tag': 'rpai'},
        # {'imu': ['rp', 'gl'], 'wifi': False, 'airpods': False, 'channel': 12, 'modality': 'imu', 'tag': 'rpgl'},
        {'imu': ['rh', 'rp'], 'wifi': False, 'airpods': True, 'channel': 18, 'modality': 'imu', 'tag': 'rhrpai'},
        {'imu': ['rh', 'rp', 'gl'], 'wifi': False, 'airpods': False, 'channel': 18, 'modality': 'imu', 'tag': 'rhrpgl'},
        # {'imu': ['rp', 'gl'], 'wifi': False, 'airpods': True, 'channel': 18, 'modality': 'imu', 'tag': 'rpglai'},
        # {'imu': ['rh', 'rp', 'gl'], 'wifi': False, 'airpods': True, 'channel': 24, 'modality': 'imu', 'tag': 'rhrpaigl'},
        # {'imu': ['rh', 'rp', 'lh', 'lp', 'gl'], 'wifi': False, 'airpods': True, 'channel': 36, 'modality': 'imu', 'tag': 'imuall'},
    ]
    day = get_day()
    tag = 'single_imu' 

    model_arc_name = 'TAD_single'
    gpu = 5

    model_str_list = [
        ('mamba', 8, 80, {'layer': 8, 'i': 1}),
        ('mamba', 8, 80, {'layer': 8, 'i': 2}),
    ]

    dataset_str_list = [
        ('WWADLDatasetMutiAll', 'XRFV2', -1, ''),
    ]
    for model_str in model_str_list:
        for dataset_str in dataset_str_list:
            dataset_name, dataset, channel, modality = dataset_str
            for receivers_to_keep in receivers_to_keep_list:
                model_name, batch_size, epoch, model_config, *others = model_str

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
