import torch
import torch.nn as nn
import numpy as np
from model.mamba.backbones import MambaBackbone
from model.mamba.necks import FPNIdentity, FPN1D
from model.TAD.embedding import Embedding
from model.mamba.loc_generators import PointGenerator
from utils.basic_config import Config


class WifiMamba_config(Config):
    """
    WifiMamba 的配置类，用于初始化模型参数。
    支持的参数格式：{num_classes}_{input_length}_{in_channels}
    """

    def __init__(self, model_set: str = '34_2048_270'):
        """
        初始化 WifiMamba 配置实例。
        :param model_set: 模型配置字符串，格式为 num_classes_input_length_in_channels
        """
        # 解析配置字符串
        num_classes, input_length, in_channels = model_set.split('_')
        self.num_classes = int(num_classes)  # 分类数
        self.input_length = int(input_length)  # 输入序列长度
        self.in_channels = int(in_channels)  # 输入通道数

        # Mamba Backbone 配置
        self.n_embd = 512  # 嵌入维度
        self.n_embd_ks = 7  # 卷积核大小
        self.arch = (3, 2, 5)  # 卷积层结构：基础卷积、stem 卷积、branch 卷积
        self.scale_factor = 2  # 下采样率
        self.with_ln = True  # 使用 LayerNorm

        # Pyramid Detection 配置
        self.priors = 128  # 初始特征点数量
        self.layer_num = 4  # 特征金字塔层数

class WifiMamba(nn.Module):
    def __init__(self, config):
        super(WifiMamba, self).__init__()

        self.embedding = Embedding(config.in_channels)

        # Mamba Backbone
        self.mamba_model = MambaBackbone(
            n_in=512,  # Must match the output of the embedding layer
            n_embd=config.n_embd,
            n_embd_ks=config.n_embd_ks,
            arch=config.arch,
            scale_factor=config.scale_factor,
            with_ln=config.with_ln
        )

        # Neck: FPNIdentity
        self.neck = FPNIdentity(
            in_channels=[config.n_embd] * (config.arch[-1] + 1),  # 输入特征通道，假设每一层的输出特征通道一致
            out_channel=config.n_embd,  # 输出特征通道数
            scale_factor=config.scale_factor,  # 下采样倍率
            with_ln=config.with_ln  # 是否使用 LayerNorm
        )

        # Point Generator
        # self.point_generator = PointGenerator(
        #     max_seq_len=config.input_length,  # 输入序列的最大长度
        #     fpn_levels=config.arch[-1] + 1,  # FPN 层级数
        #     scale_factor=config.scale_factor,  # FPN 层间缩放倍数
        #     regression_range=[                # 回归范围（示例）
        #         [0, 64],
        #         [64, 128],
        #         [128, 256],
        #         [256, 512],
        #         [512, 1024],
        #         [1024, 2048]
        #     ],

        # Priors Generation
        self.priors = []
        t = config.priors  # 初始特征点数量
        for i in range(config.layer_num):
            self.priors.append(
                torch.Tensor([[(c + 0.5) / t] for c in range(t)]).view(-1, 1)
            )

        self.num_classes = config.num_classes

        # 用 mamba 的 point_generator 来生成 priors
        # self.point_generator = self.mamba_model.point_generator

    def forward(self, x):
        # Step 1: Embedding x: (B, C, L)
        # x: (B, C, L)
        B, C, L = x.size()
        x = self.embedding(x)

        # Step 2: Generate Mask (All True for fixed length)
        batched_masks = torch.ones(B, 1, L, dtype=torch.bool).to(x.device)
        # Step 3: Backbone
        feats, masks = self.mamba_model(x, batched_masks)

        fpn_feats, fpn_masks = self.neck(feats, masks)

        # points = self.point_generator(fpn_feats)
        #
        # priors = torch.cat(points, dim=0).unsqueeze(0).to(x.device)

        print(feats.shape)
        # # Step 4: Neck
        # fpn_feats, fpn_masks = self.neck(feats, masks)
        #
        # # Step 5: Classification and Regression Heads
        # cls_logits = [head(feat) for head, feat in zip(self.cls_head, feats)]
        # reg_offsets = [head(feat) for head, feat in zip(self.reg_head, feats)]
        #
        # # Step 6: Generate Priors
        # points = self.point_generator(fpn_feats)
        # priors = torch.cat(points, dim=0).unsqueeze(0).to(reg_offsets[0].device)
        #
        # # Step 7: Reshape Outputs
        # batch_size = x.size(0)
        # loc = torch.cat([offset.view(batch_size, -1, 2) for offset in reg_offsets], dim=1)
        # conf = torch.cat([logit.view(batch_size, -1, self.num_classes) for logit in cls_logits], dim=1)
        #
        # return loc, conf, priors
        return 1


