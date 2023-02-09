# Copyright (c) 2023 unrea1 (Jie Chen, Tsinghua University)

import torch
from torch import nn

from wetts.utils import mask


class ReferenceEncoder(nn.Module):

    def __init__(
        self,
        in_dim,
        conv_channels,
        conv_kernel_size,
        conv_stride,
        gru_hidden_dim,
        out_dim,
    ) -> None:
        """
        in_dim: 80
        conv_channels: [32,32,64,64,128,128]
        conv_kernel_size: 3
        conv_stride: 2
        gru_hidden_dim: 128
        out_dim: 128
        """
        super().__init__()
        self.conv_stride = conv_stride
        conv_channels = [1] + conv_channels
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_dim,
                          out_dim,
                          conv_kernel_size,
                          conv_stride,
                          padding=max(1, (conv_kernel_size - 2) // 2)),
                nn.BatchNorm2d(out_dim),
                nn.ReLU(),
            ) for in_dim, out_dim in zip(conv_channels[:-1], conv_channels[1:])
        ])
        self.gru = nn.GRU(256 // conv_stride**(len(conv_channels) - 1) *
                          conv_channels[-1],
                          gru_hidden_dim,
                          batch_first=True)
        self.linear = nn.Sequential(nn.Linear(gru_hidden_dim, out_dim),
                                    nn.Tanh())

    def forward(self, x, x_length):
        """
        x: (b,d,t)
        x_length: (b)
        """
        x = x.unsqueeze(1)  # (b,1,d,t)

        x_mask = mask.get_content_mask(x_length).unsqueeze(1).unsqueeze(
            1)  # (b,1,1,t)
        #print(x.shape, x_mask.shape)
        for conv in self.convs:
            x = conv(x * x_mask)
            x_mask = x_mask[:, :, :, ::self.conv_stride]
            x_mask = x_mask[:, :, :, :x.size(3)]
        # x: (b,k,d',t')
        b, c, d, t = x.shape
        x = x.reshape(b, -1, t)
        x = x.permute(0, 2, 1)  # (b,t,d)
        _, h = self.gru(x)  # (1,b,d)
        x = self.linear(h.squeeze(0))
        return x