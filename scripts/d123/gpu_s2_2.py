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

    day = get_day()
    tag = 'tsse'

    model_arc_name = 'TAD_muti_tsse'
    gpu = 1

    model_str_list = [
        ('mamba', 8, 80, {'layer': 8, 'i': 1}),
        ('mamba', 8, 80, {'layer': 8, 'i': 2}),
    ]

    dataset_str_list = [
        ('WWADLDatasetMuti', 'all_30_3', (30, 270), 'wifiimu'),
    ]

    for dataset_str in dataset_str_list:
        dataset_name, dataset, channel, modality = dataset_str
        for model_str in model_str_list:
            model_name, batch_size, epoch, model_config, *others = model_str

            # 调用提取的函数
            updated_config = prepare_config(model_arc_name, dataset_str, model_str, config, gpu, day, dataset_root_path, python_path, tag)

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
