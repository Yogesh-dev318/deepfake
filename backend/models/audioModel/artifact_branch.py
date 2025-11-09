
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")

from preprocessing.audio_process import CFG


# -------------------------
# Efficient Channel Attention (ECA) module
# -------------------------
class ECA(nn.Module):
    def __init__(self, channels, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size-1)//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        y = self.avg_pool(x)
        y = y.squeeze(-1).transpose(-1,-2)
        y = self.conv(y)
        y = self.sigmoid(y).transpose(-1,-2).unsqueeze(-1)
        return x * y.expand_as(x)
    

# -------------------------
# Artifact Branch: Multi-scale Residual CNN + ECA
# -------------------------
def conv_bn_relu(in_ch, out_ch, kernel, stride=1, padding=0):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True)
    )

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = conv_bn_relu(in_ch, out_ch, kernel=3, padding=1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch)
        )
        self.activation = nn.ReLU(inplace=True)
        if in_ch != out_ch:
            self.skip = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch))
        else:
            self.skip = nn.Identity()
        self.pool = nn.MaxPool2d(2)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + self.skip(x)
        out = self.activation(out)
        out = self.pool(out)
        return out

class ArtifactsBranch(nn.Module):
    """
    Deep multi-scale CNN with ECA attention modules.
    Input: spec (freq_bins x time)
    """
    def __init__(self, in_channels=1, base_channels=CFG["artifact_base_channels"], 
                 embedding_dim=CFG["embedding_dim"]):
        super().__init__()
        # initial multi-scale conv (3x3, 5x5)
        self.ms_conv = nn.ModuleList([
            conv_bn_relu(in_channels, base_channels, kernel=3, padding=1),
            conv_bn_relu(in_channels, base_channels, kernel=5, padding=2)
        ])
        # stack of residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(base_channels, base_channels*2),
            ResidualBlock(base_channels*2, base_channels*4),
            ResidualBlock(base_channels*4, base_channels*8)
        ])
        # ECA attention modules
        self.eca1 = ECA(base_channels*2, k_size=3)
        self.eca2 = ECA(base_channels*4, k_size=3)
        self.eca3 = ECA(base_channels*8, k_size=3)

        self.global_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(base_channels*8, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, embedding_dim)
        )
    
    def forward(self, spec):
        x = spec.unsqueeze(1)
        xs = [m(x) for m in self.ms_conv]
        x = sum(xs)
        x = self.res_blocks[0](x)
        x = self.eca1(x)
        x = self.res_blocks[1](x)
        x = self.eca2(x)
        x = self.res_blocks[2](x)
        x = self.eca3(x)
        x = self.global_pool(x).view(x.size(0), -1)
        emb = self.fc(x)
        return emb
