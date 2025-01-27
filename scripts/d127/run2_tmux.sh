#!/bin/bash

# 定义 tmux 会话的名称和对应的输入参数
sessions=("run1" "run2" "run3" "run4" "run5" "run6" "run7" "run8")
params=("0" "1" "2" "3" "4" "5" "6" "7")
# sessions=("run1" "run2")
# params=("0" "1")

# 循环遍历每个会话和对应的参数
for i in ${!sessions[@]}; do
    session=${sessions[$i]}
    param=${params[$i]}

    # 使用 Ctrl+C 停止当前会话中的运行中的 Python 脚本
    tmux send-keys -t "$session" C-c

    # 等待一段时间，以确保进程被成功中断
    sleep 1

    # 在对应的 tmux 会话中运行 Python 脚本，传递不同的输入参数
    tmux send-keys -t "$session" "/root/.conda/envs/t1/bin/python /root/shared-nvme/code/WWADL_code_mac/scripts/d127/gpu2_1.py --gpu $param" C-m
done