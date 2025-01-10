import os
import numpy as np
from scipy.interpolate import interp1d
from dataset.modality.action import id_to_action

class WWADLBase():
    def __init__(self, file_path):
        self.data = None
        self.label = None
        self.file_name = os.path.basename(file_path)


    def load_data(self, file_path):
        pass

    def show_info(self):
        print(self.data.shape)
        print(self.label)

    def segment(self, time_len=30, step=3, sample_rate=200, target_len=2048):
        """
        滑动窗口切分数据并生成对应的标注 label
        Args:
            time_len: 每个窗口的时间长度（秒）
            step: 滑动窗口的步长（秒）
            sample_rate: 采样率（每秒采样点数）
        Returns:
            segmented_data: 切分后的数据 (num_windows, ...)
            targets: 每个窗口对应的目标标签
        """

        # 计算窗口长度和滑动步长
        window_len = time_len * sample_rate
        window_step = step * sample_rate

        if self.data.shape[0] < window_len:
            raise ValueError(f"Data length ({self.data.shape[0]}) is less than window length ({window_len}).")

        # 滑动窗口切分数据
        segmented_data = []
        targets = []
        # print(self.data.shape)
        for start in range(0, self.data.shape[0] - window_len + 1, window_step):
            end = start + window_len
            window_data = self.data[start:end]

            # 插值到 target_len 长度
            original_indices = np.linspace(0, window_len - 1, window_len)
            target_indices = np.linspace(0, window_len - 1, target_len)
            interpolated_data = interp1d(original_indices, window_data, axis=0, kind='linear')(target_indices)

            segmented_data.append(interpolated_data)

            # 处理 label 生成目标标签
            window_start_time = start
            window_end_time = end
            window_targets = []

            for lbl in self.label:
                obj_start_time = lbl[2]
                obj_end_time = lbl[3]
                obj_label = lbl[1]

                # 计算目标在窗口内的交集
                intersection_start = max(window_start_time, obj_start_time)
                intersection_end = min(window_end_time, obj_end_time)
                intersection_duration = max(0, intersection_end - intersection_start)
                obj_duration = obj_end_time - obj_start_time

                # 判断交集是否覆盖目标时间的 80%
                if intersection_duration / obj_duration >= 0.8:
                    # 计算相对时间位置
                    relative_start = max(0, obj_start_time - window_start_time) / window_len
                    relative_end = min(window_len, obj_end_time - window_start_time) / window_len
                    window_targets.append([relative_start, relative_end, obj_label])

            # 如果窗口内有目标，则添加到 targets
            if window_targets:
                targets.append(window_targets)
        # 转换为 np.array
        segmented_data = np.array(segmented_data)
        return segmented_data, targets

    def generate_annotations(self, subset):
        """
        根据标签数据生成JSON格式的标注文件。

        参数:
            subset (str): 数据集的子集名称，例如 'validation' 或 'training'。

        返回:
            dict: JSON格式的标注数据。
        """
        if self.label is None:
            raise ValueError("Label data is not set.")

        annotations = []
        for row in self.label:
            action_id = row[1]
            start_id = row[2]
            end_id = row[3]
            annotations.append({
                "segment": [start_id, end_id],
                "label": f"{id_to_action[action_id]}"
            })

        json_data = {
            "subset": subset,
            "annotations": annotations
        }
        return (self.file_name, json_data)
