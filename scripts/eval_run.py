import os
import sys
import json

# 定义路径
project_path = '/root/shared-nvme/code/WWADL_code_mac'
dataset_root_path = '/root/shared-nvme/dataset'
causal_conv1d_path = '/root/shared-nvme/video-mamba-suite/causal-conv1d'
mamba_path = '/root/shared-nvme/video-mamba-suite/mamba'
sys.path.append(project_path)

os.environ["PYTHONPATH"] = f"{project_path}:{causal_conv1d_path}:{mamba_path}:" + os.environ.get("PYTHONPATH", "")

import numpy as np
from strategy.evaluation.eval_detection import ANETdetection

def eval(eval_gt, eval_pr):
    # Define tIoU thresholds
    tious = np.linspace(0.5, 0.95, 10)

    # Initialize ANETdetection
    anet_detection = ANETdetection(
        ground_truth_filename=eval_gt,
        prediction_filename=eval_pr,
        subset='test',
        tiou_thresholds=tious
    )

    mAPs, average_mAP, ap = anet_detection.evaluate()
    print(average_mAP)


if __name__ == '__main__':

    eval_gt = '/root/shared-nvme/dataset/XRFV2/imu_annotations.json'
    eval_pr = '/root/shared-nvme/code_result/result/25_01-23/fusion_grc/WWADLDatasetMuti_all_30_3_mamba_layer_8_i_1/checkpoint_mamba_mamba_layer_8_i_1-epoch-79.pt.json'
    # eval_pr = '/root/shared-nvme/code_result/result/25_01-23/fusion_grc/WWADLDatasetMuti_all_30_3_mamba_layer_8_i_2/checkpoint_mamba_mamba_layer_8_i_2-epoch-76.pt.json'
    # eval_pr = '/root/shared-nvme/code_result/result/25_01-23/fusion_tsse/WWADLDatasetMuti_all_30_3_mamba_layer_8_i_1/checkpoint_mamba_mamba_layer_8_i_1-epoch-79.pt.json'
    eval(eval_gt, eval_pr)