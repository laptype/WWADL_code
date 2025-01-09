import sys
import os
import torch

# project_path = '/home/lanbo/WWADL/WWADL_code'
# dataset_root_path = '/data/WWADL/dataset'

project_path = '/root/shared-nvme/code/WWADL_code'
dataset_root_path = '/root/shared-nvme/dataset'

sys.path.append(project_path)

from utils.setting import get_day, get_time, write_setting, get_result_path, get_log_path, Run_config
from global_config import config



if __name__ == '__main__':

    day = get_day()

    model_str_list = [
        # model,    batch size,      epoch
        ('wifiTAD', 32, 55)
    ]

    dataset_str_list = [

        ('WWADLDatasetSingle', 'wifi_30_3_0', '34_2048_90_0'),
        # ('WWADLDatasetSingle', 'wifi_30_3_1', '34_2048_90_0'),
        # ('WWADLDatasetSingle', 'wifi_30_3_2', '34_2048_90_0'),
        ('WWADLDatasetSingle', 'imu_30_3_gl', '34_2048_6_0'),
        # # ('WWADLDatasetSingle', 'imu_30_3_lh', '34_2048_6_0'),
        ('WWADLDatasetSingle', 'imu_30_3_rh', '34_2048_6_0'),
        # # ('WWADLDatasetSingle', 'imu_30_3_lp', '34_2048_6_0'),
        ('WWADLDatasetSingle', 'imu_30_3_rp', '34_2048_6_0'),
        #
        ('WWADLDatasetSingle', 'wifi_30_3', '34_2048_270_0'),
        ('WWADLDatasetSingle', 'imu_30_3', '34_2048_30_0'),

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
            config["training"]["DDP"]["devices"] = [0, 1]

            config["model"]["model_set"] = model_set

            config["training"]["lr_rate"] = 2e-05

            test_gpu = 0

            # TAG ===============================================================================================
            tag = f'device'

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
            os.system(
                f"CUDA_VISIBLE_DEVICES={run.ddp_devices} {run.python_path} "
                f"{run.main_path} --is_train true --config_path {run.config_path}"
            )

            config['endtime'] = get_time()
            write_setting(config, os.path.join(config['path']['result_path'], 'setting.json'))

            # TEST ==============================================================================================
            os.system(
                f"CUDA_VISIBLE_DEVICES={test_gpu} {run.python_path} "
                f"{run.main_path} --config_path {run.config_path}"
            )
