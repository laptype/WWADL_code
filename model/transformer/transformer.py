import torch
import torch.nn.init as init
import torch.nn as nn
from utils.basic_config import Config
from model.mamba.necks import FPNIdentity
from model.TAD.embedding import Embedding
from model.TAD.head import PredictionHead
from model.TAD.module import ScaleExp
from model.transformer.backbones import ConvTransformerBackbone
from model.models import register_model, register_model_config

@register_model_config('Transformer')
class Transformer_config(Config):
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

        layer = int(self.additional_config.get('l', 4))

        # Backbone 配置
        self.n_embd = 512
        self.n_head = 8
        self.n_embd_ks = 3  # 卷积核大小
        self.arch = (2, layer, 4)  # 卷积层结构：基础卷积、stem 卷积、branch 卷积
        # print(f'self.arch: {self.arch}')
        self.max_len = 256
        # window size for self attention; <=1 to use full seq (ie global attention)
        n_mha_win_size = -1
        self.mha_win_size = [n_mha_win_size] * (1 + self.arch[-1])
        self.scale_factor = 2
        self.with_ln= True
        self.attn_pdrop = 0.0
        self.proj_pdrop = 0.4
        self.path_pdrop = 0.1
        self.use_abs_pe = False
        self.use_rel_pe = False

        self.priors = 256  # 初始特征点数量
        self.layer_skip = 3

@register_model('Transformer')
class Transformer(nn.Module):
    def __init__(self, config: Transformer_config):
        super(Transformer, self).__init__()

        self.embedding = Embedding(config.in_channels, stride=2)

        # Transformer Backbone
        self.backbone = ConvTransformerBackbone(
            n_in=512,  # Must match the output of the embedding layer
            n_embd=config.n_embd,
            n_head=config.n_head,
            n_embd_ks=config.n_embd_ks,
            arch=config.arch,
            max_len=config.max_len,
            mha_win_size=config.mha_win_size,
            scale_factor=config.scale_factor,
            with_ln=config.with_ln,
            attn_pdrop=config.attn_pdrop,
            proj_pdrop=config.proj_pdrop,
            path_pdrop=config.path_pdrop,
            use_abs_pe=config.use_abs_pe,
            use_rel_pe=config.use_rel_pe
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
        for m in self.backbone.modules():
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

        B, C, L = x.size()
        # Step 2: Generate Mask (All True for fixed length)
        batched_masks = torch.ones(B, 1, L, dtype=torch.bool).to(x.device)
        # Step 3: Backbone
        feats, masks = self.backbone(x, batched_masks)

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

