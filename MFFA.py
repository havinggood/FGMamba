#   hidden_states_list每个hidden_states形状为[B, 384, 320]

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from .FDConv import FDConv
# Multi-stage Feature Aggregation (MFA)



class MFFA(nn.Module):
    def __init__(self, embed_dim, num_encoding_strategies):
        super(MFFA, self).__init__()
        self.embed_dim = embed_dim
        self.num_encoding_strategies = num_encoding_strategies
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, num_encoding_strategies * embed_dim)
        )
        self.fdconv = FDConv(
            in_channels=384,  # 输入通道数与图像通道数一致（默认3）
            out_channels=384,  # 输出通道数保持相同
            kernel_size=3,  # 卷积核大小（根据需求调整）
            padding=1,  # 保持空间分辨率
            stride=1,
            kernel_num=32,
        )

    def forward(self, hidden_states_list):
        f0, f1, f2, f3, f4, f5 = hidden_states_list[:6]  # 每个形状 [B, 384, 320]
        F0 = f0 + f1 + f2 + f3 + f4 + f5
        f0 = f0.view(f0.size(0), -1, 20, 16)
        f1 = f1.view(f1.size(0), -1, 20, 16)
        f2 = f2.view(f2.size(0), -1, 20, 16)
        f3 = f3.view(f3.size(0), -1, 20, 16)
        f4 = f4.view(f4.size(0), -1, 20, 16)
        f5 = f5.view(f5.size(0), -1, 20, 16)

        F = f0 + f1 + f2 + f3 + f4 + f5 # [B, 384, 20, 16]
        F = self.fdconv(F)
        # GAP
        g = F.mean(dim=(2, 3))  #  [B, 384]
        m = self.mlp(g) # [B, 384*6] -> [B, 2304]
        h = m.view(-1, self.num_encoding_strategies, self.embed_dim)    # [B, 6, 384]
        p = torch.softmax(h, dim=1) # [B, 6, 384]
        x0_f, x1_f, x2_f, x3_f, x4_f, x5_f = torch.split(p, 1, dim=1)   # each shape: [B, 1, 384]
        x0_p, x1_p, x2_p, x3_p, x4_p, x5_p = (
            x0_f.squeeze(1), x1_f.squeeze(1), x2_f.squeeze(1),
            x3_f.squeeze(1), x4_f.squeeze(1), x5_f.squeeze(1)
        )  # each shape: [batch_size, embed_dim]
        V = (x0_p[:, :, None, None] * f0 +  # V.shape = [B, 384, 20, 16]
             x1_p[:, :, None, None] * f1 +
             x2_p[:, :, None, None] * f2 +
             x3_p[:, :, None, None] * f3 +
             x4_p[:, :, None, None] * f4 +
             x5_p[:, :, None, None] * f5)  # Sum up to x5
        V0 = V.view(V.size(0), V.size(1), -1)
        output = F0 + V0    # res
        return output
