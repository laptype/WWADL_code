import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
import torch.nn.init as init
from model.mamba.backbones import MambaBackbone
from model.mamba.necks import FPNIdentity, FPN1D
from model.TAD.embedding import Embedding
from model.mamba.loc_generators import PointGenerator
from utils.basic_config import Config
from model.mamba.head import PtTransformerClsHead, PtTransformerRegHead
from model.mamba.downsample import Downsample
from model.TAD.head import PredictionHead
from model.TAD.module import ScaleExp
from model.TAD.backbone import TSSE, LSREF
from model.models import register_model, register_model_config

@register_model_config('WifiMambaSkip')
class WifiMambaSkip_config(Config):
    """
    WifiMamba 的配置类，用于初始化模型参数。
    支持的参数格式：{num_classes}_{input_length}_{in_channels}
    """
    def __init__(self, model_set: str = '34_2048_270_l-3_m-dbm'):
        """
        初始化 WifiMamba 配置实例。
        :param model_set: 模型配置字符串，格式为 num_classes_input_length_in_channels
        """
        # 解析配置字符串
        num_classes, input_length, in_channels, *others = model_set.split('_')
        self.num_classes = int(num_classes)  # 分类数
        self.input_length = int(input_length)  # 输入序列长度
        self.in_channels = int(in_channels)  # 输入通道数

        # 解析 others 部分（如果有）
        self.additional_config = {}
        if others:
            for item in others:
                if '-' in item:
                    key, value = item.split('-')
                    self.additional_config[key] = str(value)  # 假设 value 是整数，可以根据需求修改类型

        layer = 4

        if 'l' in self.additional_config:
            layer = int(self.additional_config['l'])

        # Mamba Backbone 配置
        self.n_embd = 512  # 嵌入维度
        self.n_embd_ks = 3  # 卷积核大小

        self.arch = (2, layer, 4)  # 卷积层结构：基础卷积、stem 卷积、branch 卷积
        print(f'self.arch: {self.arch}')
        self.scale_factor = 2  # 下采样率
        self.with_ln = True  # 使用 LayerNorm

        # Pyramid Detection 配置
        self.priors = 256  # 初始特征点数量
        # self.layer_num = 4  # 特征金字塔层数

        # self.head = 'mamba_head'
        self.head = 'wifiadl_head'

        self.layer_skip = 3
        self.mamba_type = 'dbm'
        # vim
        if 'm' in self.additional_config:
            self.mamba_type = self.additional_config['m']



@register_model('WifiMambaSkip')
class WifiMambaSkip(nn.Module):
    def __init__(self, config: WifiMambaSkip_config):
        super(WifiMambaSkip, self).__init__()

        self.embedding = Embedding(config.in_channels, stride=2)
        self.train_center_sample = 'radius'
        self.train_center_sample_radius = 1.5
        # Mamba Backbone
        self.mamba_model = MambaBackbone(
            n_in=512,  # Must match the output of the embedding layer
            n_embd=config.n_embd,
            n_embd_ks=config.n_embd_ks,
            arch=config.arch,
            scale_factor=config.scale_factor,
            with_ln=config.with_ln,
            mamba_type=config.mamba_type
        )

        # Neck: FPNIdentity
        self.neck = FPNIdentity(
            in_channels=[config.n_embd] * (config.arch[-1] + 1),  # 输入特征通道，假设每一层的输出特征通道一致
            out_channel=config.n_embd,  # 输出特征通道数
            scale_factor=config.scale_factor,  # 下采样倍率
            with_ln=config.with_ln  # 是否使用 LayerNorm
        )

        # self.skip_tsse = nn.ModuleList()
        # for i in range(config.layer_skip):
        #     self.skip_tsse.append(TSSE(in_channels=512, out_channels=256, kernel_size=3, stride=2, length=(config.input_length // 2)//(2**i)))

        if config.head == 'mamba_head':

            # Classification Head
            self.cls_head = PtTransformerClsHead(
                input_dim=config.n_embd,  # FPN 输出特征通道数
                feat_dim=256,  # 分类头的中间层特征维度
                num_classes=config.num_classes,  # 分类类别数
                prior_prob=0.01,  # 初始化概率
                num_layers=3,  # 卷积层数
                kernel_size=3,  # 卷积核大小
                act_layer=nn.ReLU,  # 激活函数
                with_ln=config.with_ln,  # 是否使用 LayerNorm
                empty_cls=[]  # 空类别
            )

            # Regression Head
            self.reg_head = PtTransformerRegHead(
                input_dim=config.n_embd,  # FPN 输出特征通道数
                feat_dim=256,  # 回归头的中间层特征维度
                fpn_levels=config.arch[-1] + 1,  # FPN 层级数
                num_layers=3,  # 卷积层数
                kernel_size=3,  # 卷积核大小
                act_layer=nn.ReLU,  # 激活函数
                with_ln=config.with_ln  # 是否使用 LayerNorm
            )
        else:
            self.PredictionHead = PredictionHead()
            self.loc_heads = nn.ModuleList()

        self.num_classes = config.num_classes
        self.total_frames = config.input_length

        self.priors = []
        t = config.priors
        for i in range(config.arch[-1] + 1):
            self.loc_heads.append(ScaleExp())
            self.priors.append(
                torch.Tensor([[(c + 0.5) / t] for c in range(t)]).view(-1, 1)
            )
            t = t // 2
        # Initialize weights
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize embedding
        for m in self.embedding.modules():
            if isinstance(m, nn.Conv1d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)

        # Initialize backbone
        for m in self.mamba_model.modules():
            if isinstance(m, nn.Conv1d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)

        # Initialize neck
        for m in self.neck.modules():
            if isinstance(m, nn.Conv1d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)


        # Initialize LayerNorm
        for m in self.modules():
            if isinstance(m, nn.LayerNorm):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)


    def forward(self, x):
        # Step 1: Embedding x: (B, C, L)
        # x: (B, C, 2048)
        B, C, L = x.size()
        x = self.embedding(x)
        # x: (B, 512, 256)

        # resnet TODO 不用这个试试, stride 改成 2
        '''
            1. x = self.embedding(x, stride=2)
            2. vit
                x = self._pickup_patching(x)    # 16, 63, 1440
                x = self.embedding(x)   # 16 63 512
        '''
        # for i in range(len(self.skip_tsse)):
        #     x = self.skip_tsse[i](x)

        # x: (B, 512, 256)

        B, C, L = x.size()

        # Step 2: Generate Mask (All True for fixed length)
        batched_masks = torch.ones(B, 1, L, dtype=torch.bool).to(x.device)
        # Step 3: Backbone
        feats, masks = self.mamba_model(x, batched_masks)

        fpn_feats, fpn_masks = self.neck(feats, masks)


        out_offsets = []
        out_cls_logits = []

        for i, feat in enumerate(fpn_feats):
            assert not torch.isnan(feat).any(), "NaN detected in loc_logits before PredictionHead"
            loc_logits, conf_logits = self.PredictionHead(feat)
            assert not torch.isnan(loc_logits).any(), f"NaN detected in loc_logits at layer {i}"
            out_offsets.append(
                self.loc_heads[i](loc_logits)
                    .view(B, 2, -1)
                    .permute(0, 2, 1).contiguous()
            )
            out_cls_logits.append(
                conf_logits.view(B, self.num_classes, -1)
                    .permute(0, 2, 1).contiguous()
            )

        priors = torch.cat(self.priors, 0).to(x.device).unsqueeze(0)
        loc = torch.cat([o.view(B, -1, 2) for o in out_offsets], 1)
        conf = torch.cat([o.view(B, -1, self.num_classes) for o in out_cls_logits], 1)

        return {
            'loc': loc,
            'conf': conf,
            'priors': priors # trainer ddp需要弄成priors[0]
        }


