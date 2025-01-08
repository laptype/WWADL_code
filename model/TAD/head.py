import torch
import torch.nn as nn
from model.TAD.module import Unit1D

class Tower(nn.Module):
    def __init__(self, out_channels, layer):
        super().__init__()
        
        conf_towers = [] 
        for i in range(layer):
            conf_towers.append(
                nn.Sequential(
                    Unit1D(
                        in_channels=out_channels,
                        output_channels=out_channels,
                        kernel_shape=3,
                        stride=1,
                        use_bias=True,
                        activation_fn=None
                    ),
                    nn.GroupNorm(32, out_channels),
                    nn.ReLU(inplace=True)
                )
            )
            
        self.conf_tower = nn.Sequential(*conf_towers)

    def forward(self, x):
        return self.conf_tower(x)


class conf_head(nn.Module):
    def __init__(self, out_channels=512, num_classes=34):
        super().__init__()
        self.conf = Unit1D(
            in_channels=out_channels,
            output_channels=num_classes,
            kernel_shape=3,
            stride=1,
            use_bias=True,
            activation_fn=None
        )

    def forward(self, x):
        x = self.conf(x)
        return x


class loc_head(nn.Module):
    def __init__(self, out_channels=512):
        super().__init__()
        self.loc =nn.Sequential(Unit1D(
                in_channels=out_channels,
                output_channels=out_channels,
                kernel_shape=3,
                stride=1,
                use_bias=True,
                activation_fn=None
            ),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
        Unit1D(
                in_channels=out_channels,
                output_channels=2,
                kernel_shape=3,
                stride=1,
                use_bias=True,
                activation_fn=None
            ))

    def forward(self, x):
        x = self.loc(x)
        return x


class PredictionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.loc_tower = Tower(512, 3)
        self.conf_tower = Tower(512, 3)
        
        self.loc_head = loc_head()
        self.conf_head = conf_head()
        
    def forward(self, x):
        # 获取特征
        loc_feat = self.loc_tower(x)
        conf_feat = self.conf_tower(x)

        # 打印中间特征形状
        # print(f"loc_feat shape: {loc_feat.shape}")
        # print(f"conf_feat shape: {conf_feat.shape}")

        # 通过预测头
        loc_feat = self.loc_head(loc_feat)
        conf_feat = self.conf_head(conf_feat)

        # 打印最终形状
        # print(f"loc_feat after loc_head: {loc_feat.shape}")
        # print(f"conf_feat after conf_head: {conf_feat.shape}")

        return loc_feat, conf_feat