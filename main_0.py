import os
import sys
import torch

project_path = '/home/lanbo/WWADL/WWADL_code'
sys.path.append(project_path)

import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import json
import multiprocessing
from training import train

import logging

def setup_logging(log_file_path):
    """
    设置日志配置，将日志内容输出到控制台和文件。
    Args:
        log_file_path (str): 日志文件的路径，例如 'output.log'。
    """
    # 创建日志目录（如果不存在）
    log_dir = os.path.dirname(log_file_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger()  # 获取根日志记录器
    logger.setLevel(logging.INFO)

    # 检查是否已经设置了处理器，避免重复
    if not logger.handlers:
        # 文件处理器
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(logging.INFO)

        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # 日志格式
        formatter = logging.Formatter('%(asctime)s - %(filename)s - %(levelname)s: %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # 添加处理器到记录器
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

# 初始化每个进程的分布式环境
def setup(rank, world_size):
    dist.init_process_group(
        backend="nccl",  # 如果使用 GPU
        init_method="env://",  # 通过环境变量初始化
        rank=rank,
        world_size=world_size
    )
    torch.cuda.set_device(rank)  # 为每个进程设置 GPU

# 清理分布式环境
def cleanup():
    dist.destroy_process_group()

# 分布式训练主逻辑
def distributed_train(rank, world_size, config):
    # 初始化分布式环境
    setup(rank, world_size)

    # 启动训练
    train(config=config)

    # 清理分布式环境
    cleanup()

# 设置可见 GPU
def set_visible_gpus(gpu_ids):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
    print(f"Using GPUs: {gpu_ids}")

def set_master_addr_and_port(master_addr="127.0.0.1", master_port="29500"):
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port

# 主入口函数
def main(config, gpu_ids, nproc_per_node, is_train):
    # 初始化命令行参数
    set_master_addr_and_port()
    print(config['path']['log_path']['train'])
    setup_logging(config['path']['log_path']['train'])
    logging.info("Starting main process.")

    os.environ["RANK"] = "0"     # 当前进程的全局 rank（通常从 0 开始）
    os.environ["WORLD_SIZE"] = f"{nproc_per_node}"  # 全局进程数
    os.environ["LOCAL_RANK"] = "0"  # 当前进程在本机的 rank

    if is_train:
        # 设置可见 GPU
        set_visible_gpus(gpu_ids)

        # 获取 GPU 数量
        world_size = nproc_per_node

        # 使用 torch.multiprocessing.spawn 启动多进程
        mp.spawn(
            distributed_train,
            args=(world_size, config),
            nprocs=world_size,
            join=True
        )
    else:
        print("Testing is not implemented yet.")



# 允许通过脚本调用或从其他模块调用
if __name__ == '__main__':
    from global_config import config
    main(config,
         gpu_ids='1,2',
         nproc_per_node=2,
         is_train=True)