import torch
import torch.nn as nn
import numpy as np
from model.TAD.embedding import Embedding
from model.TAD.module import ScaleExp
from model.TAD.backbone import TSSE, LSREF
from model.TAD.head import PredictionHead
from utils.basic_config import Config

class wifiTAD_config(Config):

    def __init__(self, model_set: str = '34_2048_30'):
        """
        Initialize the wifiTAD_config instance with num_classes and input_length.
        Format: {num_classes}_{input_length}_{in_channels}
        """
        num_classes, input_length, in_channels = model_set.split('_')
        self.num_classes = int(num_classes)
        self.input_length = int(input_length)
        self.layer_num = 3
        self.skip_ds_layer = 3
        self.priors = 128
        self.in_channels = int(in_channels)
        print(f'self.in_channels: {self.in_channels}')

def wifiTAD(config: wifiTAD_config):
    net = wifitad(config)
    return net

class Pyramid_Detection(nn.Module):
    def __init__(self, skip_ds_layer, layer_num, input_length, priors, num_classes):
        super(Pyramid_Detection, self).__init__()
        self.layer_skip = skip_ds_layer
        self.skip_tsse = nn.ModuleList()

        self.num_classes = num_classes
        
        self.layer_num = layer_num
        self.PyTSSE = nn.ModuleList()
        self.PyLSRE = nn.ModuleList()
        self.loc_heads = nn.ModuleList()
        
        for i in range(self.layer_skip):
            self.skip_tsse.append(TSSE(in_channels=512, out_channels=256, kernel_size=3, stride=2, length=(input_length // 2)//(2**i)))
        
        
        for i in range(layer_num):
            self.PyTSSE.append(TSSE(in_channels=512, out_channels=256, kernel_size=3, stride=2, length=priors//(2**i)))
            self.PyLSRE.append(LSREF(len=priors//(2**i),r=((input_length // 2)//priors)*(2**i)))
            
        self.PredictionHead = PredictionHead()
        self.priors = []
        t = priors
        for i in range(layer_num):
            self.loc_heads.append(ScaleExp())
            self.priors.append(
                torch.Tensor([[(c + 0.5) / t] for c in range(t)]).view(-1, 1)
            )
            t = t // 2
        
    def forward(self, embedd):
        
        deep_feat = embedd
        global_feat = embedd.detach()
        for i in range(len(self.skip_tsse)):
            deep_feat = self.skip_tsse[i](deep_feat)
        
        
        batch_num = deep_feat.size(0)
        out_feats = []
        locs = []
        confs = []
        for i in range(self.layer_num):
            deep_feat = self.PyTSSE[i](deep_feat)
            out = self.PyLSRE[i](deep_feat, global_feat)
            out_feats.append(out)
        
        for i, feat in enumerate(out_feats):
            loc_logits, conf_logits = self.PredictionHead(feat)
            locs.append(
                self.loc_heads[i](loc_logits)
                    .view(batch_num, 2, -1)
                    .permute(0, 2, 1).contiguous()
            )
            confs.append(
                conf_logits.view(batch_num, self.num_classes, -1)
                    .permute(0, 2, 1).contiguous()
            )

        loc = torch.cat([o.view(batch_num, -1, 2) for o in locs], 1)
        conf = torch.cat([o.view(batch_num, -1, self.num_classes) for o in confs], 1)
        priors = torch.cat(self.priors, 0).to(loc.device).unsqueeze(0)
        return loc, conf, priors


class wifitad(nn.Module):
    def __init__(self, config: wifiTAD_config):
        super(wifitad, self).__init__()
        self.embedding = Embedding(config.in_channels)
        self.pyramid_detection = Pyramid_Detection(
            skip_ds_layer=config.skip_ds_layer,
            layer_num=config.layer_num,
            input_length=config.input_length,
            priors=config.priors,
            num_classes=config.num_classes
        )
        self.reset_params()
        
    @staticmethod
    def weight_init(m):
        def glorot_uniform_(tensor):
            fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
            scale = 1.0
            scale /= max(1., (fan_in + fan_out) / 2.)
            limit = np.sqrt(3.0 * scale)
            return nn.init._no_grad_uniform_(tensor, -limit, limit)

        if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d) \
                or isinstance(m, nn.ConvTranspose3d):
            glorot_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def reset_params(self):
        for _, m in enumerate(self.modules()):
            self.weight_init(m)
    
    def forward(self, x):
        x = self.embedding(x)

        loc, conf, priors = self.pyramid_detection(x)

        # print(priors.shape)

        return {
            'loc': loc,
            'conf': conf,
            'priors': priors[0]
        }
