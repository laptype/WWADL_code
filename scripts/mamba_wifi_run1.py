import sys
import os
import torch
import subprocess

# project_path = '/home/lanbo/WWADL/WWADL_code'
# dataset_root_path = '/data/WWADL/dataset'

# 定义路径
project_path = '/root/shared-nvme/code/WWADL_code'
dataset_root_path = '/root/shared-nvme/dataset'
causal_conv1d_path = '/root/shared-nvme/causal-conv1d'
mamba_path = '/root/shared-nvme/video-mamba-suite/mamba'
sys.path.append(project_path)
os.environ["PYTHONPATH"] = f"{project_path}:{causal_conv1d_path}:{mamba_path}:" + os.environ.get("PYTHONPATH", "")


from utils.setting import get_day, get_time, write_setting, get_result_path, get_log_path, Run_config
from global_config import config



if __name__ == '__main__':

    day = get_day()

    model_str_list = [
        # model,    batch size,      epoch
        ('WifiMambaSkip', 16, 55),
        # ('WifiMamba', 16, 55),
        # ('wifiTAD', 16, 55)
    ]

    dataset_str_list = [

        ('WWADLDatasetSingle', 'wifi_30_3', '34_2048_270_l-8_m-dbm'),
        ('WWADLDatasetSingle', 'wifi_30_3', '34_2048_270_l-8_m-vim'),
        # ('WWADLDatasetSingle', 'imu_30_3', '34_2048_30_l-8'),
        # ('WWADLDatasetSingle', 'wifi_30_3', '34_2048_270_l-4'),
        # ('WWADLDatasetSingle', 'imu_30_3', '34_2048_30_l-4'),
        # ('WWADLDatasetSingle', 'wifi_30_3', '34_2048_270_l-4'),
        # ('WWADLDatasetSingle', 'wifi_30_3', '34_2048_270_1'),
        # ('WWADLDatasetSingle', 'imu_30_3', '34_2048_30_0'),

        # ('WWADLDatasetSingle', 'wifi_30_3_0', '34_2048_90_0'),
        # ('WWADLDatasetSingle', 'wifi_30_3_1', '34_2048_90_0'),
        # ('WWADLDatasetSingle', 'wifi_30_3_2', '34_2048_90_0'),
        # # ('WWADLDatasetSingle', 'imu_30_3_gl', '34_2048_6_0'),
        # ('WWADLDatasetSingle', 'imu_30_3_lh', '34_2048_6_0'),
        # # ('WWADLDatasetSingle', 'imu_30_3_rh', '34_2048_6_0'),
        # ('WWADLDatasetSingle', 'imu_30_3_lp', '34_2048_6_0'),
        # ('WWADLDatasetSingle', 'imu_30_3_rp', '34_2048_6_0'),


        #
        # ('WWADLDatasetSingle', 'wifi_30_3', '34_2048_270_1'),
        # ('WWADLDatasetSingle', 'imu_30_3', '34_2048_30_1'),

        # ('WWADLDatasetSingle', 'wifi_30_3', '34_2048_270_1'),
        # ('WWADLDatasetSingle', 'imu_30_3', '34_2048_30_1'),
        # ('WWADLDatasetSingle', 'wifi_30_3', '34_2048_270_0'),
        # ('WWADLDatasetSingle', 'imu_30_3', '34_2048_30_0'),
    ]

    for dataset_str in dataset_str_list:
        dataset_name, dataset, model_set = dataset_str
        for model_str in model_str_list:
            model_name, batch_size, epoch = model_str

            config['datetime'] = get_time()
            config["training"]["DDP"]["enable"] = True
            config["training"]["DDP"]["devices"] = [0]

            config["model"]["model_set"] = model_set
            config["model"]["backbone_name"] = model_name

            config["training"]["lr_rate"] = 4e-05

            test_gpu = 0

            # TAG ===============================================================================================
            tag = f'mambaskip_dbm'

            config['path']['dataset_path'] = os.path.join(dataset_root_path, dataset)
            config['path']['log_path']      = get_log_path(config, day, f'{dataset_name}_{dataset}', model_set, tag)
            config['path']['result_path']   = get_result_path(config, day, f'{dataset_name}_{dataset}', model_set, tag)

            config['dataset']['dataset_name'] = os.path.join(dataset_name)
            config['dataset']['clip_length'] = 1500

            config["training"]['num_epoch'] = epoch
            config["training"]['train_batch_size'] = batch_size

            write_setting(config, os.path.join(config['path']['result_path'], 'setting.json'))

            # TRAIN =============================================================================================
            run = Run_config(config, 'train')

            # os.system(
            #     f"CUDA_VISIBLE_DEVICES={run.ddp_devices} {run.python_path} -m torch.distributed.launch --nproc_per_node {run.nproc_per_node} "
            #     f"--master_port='29501' --use_env "
            #     f"{run.main_path} --is_train true --config_path {run.config_path}"
            # )
            # TRAIN =============================================================================================
            train_command = (
                f"CUDA_VISIBLE_DEVICES={run.ddp_devices} {run.python_path} "
                f"{run.main_path} --is_train true --config_path {run.config_path}"
            )

            # 执行训练命令并等待其完成
            train_process = subprocess.run(train_command, shell=True)

            # 检查训练命令是否正常结束
            if train_process.returncode == 0:  # 正常结束返回 0
                config['endtime'] = get_time()
                write_setting(config, os.path.join(config['path']['result_path'], 'setting.json'))

                # TEST ==========================================================================================
                test_command = (
                    f"CUDA_VISIBLE_DEVICES={test_gpu} {run.python_path} "
                    f"{run.main_path} --config_path {run.config_path}"
                )

                # 启动测试命令
                subprocess.run(test_command, shell=True)
            else:
                print("Training process failed. Test process will not start.")
