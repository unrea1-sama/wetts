# Copyright (c) 2023 unrea1 (Jie Chen, Tsinghua University)

import torch
from torch import nn

from wetts.models.am.fastspeech2.module import reference_encoder


class EmotionEmbeddingNetwork(nn.Module):

    def __init__(self, in_dim, ref_enc_conv_channels, ref_enc_conv_kernel_size,
                 ref_enc_conv_stride, ref_enc_gru_hidden_dim, ref_enc_out_dim,
                 linear_dim, n_emo_class) -> None:
        """
        in_dim: 80
        ref_enc_conv_channels: [32,32,64,64,128,128]
        ref_enc_conv_kernel_size: 3
        ref_enc_conv_stride: 2
        ref_enc_gru_hidden_dim: 128
        ref_enc_out_dim: 128
        linear_dim: [128,256,256]
        n_emo_class: 5
        """

        super().__init__()
        self.linear_pre = nn.Linear(in_dim, 256)
        self.ref_enc = reference_encoder.ReferenceEncoder(
            in_dim, ref_enc_conv_channels, ref_enc_conv_kernel_size,
            ref_enc_conv_stride, ref_enc_gru_hidden_dim, ref_enc_out_dim)
        self.linear1 = nn.Sequential(nn.Linear(ref_enc_out_dim, linear_dim[0]),
                                     nn.ReLU())
        self.linear2 = nn.Sequential(nn.Linear(linear_dim[0], linear_dim[1]),
                                     nn.ReLU())
        self.linear3 = nn.Sequential(nn.Linear(linear_dim[1], linear_dim[2]),
                                     nn.ReLU())
        self.linear4 = nn.Linear(linear_dim[2], n_emo_class)

    def forward(self, x, x_length):
        """
        x: (b,t,d)
        x_length: (b)
        """
        x = self.linear_pre(x)
        x = self.ref_enc(x.permute(0, 2, 1), x_length)  # (b,d)
        emotion_emb = self.linear2(self.linear1(x))
        emotion_logit = self.linear4(self.linear3(emotion_emb))
        return emotion_emb, emotion_logit
