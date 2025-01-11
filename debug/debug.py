import os
import torch
from main import load_setting
from model import wifiTAD, wifiTAD_config, WifiMamba_config, WifiMamba


class Debugger:
    def __init__(self, root_path):
        self.root_path = root_path

        # 加载配置文件
        config_path = os.path.join(root_path, 'setting.json')
        self.config = load_setting(config_path)

        # 设置调试路径
        self.debug_path = os.path.join(self.root_path, "debug")
        self.data_save_path = os.path.join(self.debug_path, "error_data.pt")
        self.model_path = os.path.join(self.debug_path, "error_model.pth")

        # 加载输入数据
        self.data = torch.load(self.data_save_path)
        print(f"Input data loaded from: {self.data_save_path}")

    def load_model(self, model_class, config, device):
        try:
            # 加载模型并移动到指定设备
            model = model_class(config)
            model.load_state_dict(torch.load(self.model_path, map_location=device))
            model.to(device)
            print(f"Model parameters loaded from: {self.model_path} and moved to {device}")
            return model
        except Exception as load_error:
            print(f"An error occurred while loading the model: {load_error}")
            return None

# 示例代码：使用 Debugger
root_path = '/root/shared-nvme/code_result/result/25_01-11/testwifi/WWADLDatasetSingle_wifi_30_3_1_34_2048_90_0'
device = torch.device('cuda:1')  # 设置设备为 GPU 1

# 初始化 Debugger
debugger = Debugger(root_path)

# 将数据移动到 GPU
debugger.data = debugger.data.to(device)
print(f"Data moved to {device}: {debugger.data.shape}")

# 加载模型配置
cfg = wifiTAD_config(debugger.config["model"]["model_set"])

# 加载模型并移动到 GPU
model = debugger.load_model(wifiTAD, cfg, device)

# 前向传播
if model is not None:
    model.eval()  # 切换到评估模式
    with torch.no_grad():
        output = model(debugger.data)
        assert not torch.isnan(output['loc']).any(), "NaN detected in output_dict['loc']"
        print("Model output:", output)