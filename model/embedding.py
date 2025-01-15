import torch
import torch.nn as nn
import torch.nn.init as init
from model.TAD.embedding import Embedding
from model.TAD.backbone import TSSE

class TADEmbedding(nn.Module):
    def __init__(self, in_channels, layer, input_length):
        super(TADEmbedding, self).__init__()
        self.embedding = Embedding(in_channels)

        self.skip_tsse = nn.ModuleList()
        for i in range(layer):
            self.skip_tsse.append(TSSE(in_channels=512, out_channels=256, kernel_size=3, stride=2, length=(input_length // 2)//(2**i)))


    def initialize_weights(self):
        # Initialize embedding
        for m in self.embedding.modules():
            if isinstance(m, nn.Conv1d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.embedding(x)
        for i in range(len(self.skip_tsse)):
            x = self.skip_tsse[i](x)

        return x