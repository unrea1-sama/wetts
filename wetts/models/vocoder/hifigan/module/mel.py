# Copyright (c) 2022 Tsinghua University(Jie Chen)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import librosa
import torch
from torch import nn

from wetts.dataset import feats


class MelspectrogramLayer(nn.Module):

    def __init__(self, sr, n_fft, hop_length, win_length, n_mels, fmin, fmax):
        super().__init__()
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.fmin = fmin
        self.fmax = fmax
        # mel_basis: (n_mel, n_fft+1)
        self.mel_basis = nn.Parameter(torch.from_numpy(
            librosa.filters.mel(sr=sr,
                                n_fft=n_fft,
                                n_mels=n_mels,
                                fmin=fmin,
                                fmax=fmax)),
                                      requires_grad=False)
        self.window = nn.Parameter(torch.hann_window(win_length),
                                   requires_grad=False)

    def forward(self, x):
        """Calculating melspectrogram using torch.

        Args:
            x (torch.Tensor): input wav signal of shape (b,t).

        Returns:
            torch.Tensor: melspectrogram of shape (b,n_mels,num_frames)
        """
        return feats.torch_melspectrogram(x, self.n_fft, self.hop_length,
                                          self.win_length, self.window,
                                          self.mel_basis)