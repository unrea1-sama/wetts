# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#               2022 Tsinghua University (Jie Chen)
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
# Modified from espnet(https://github.com/espnet/espnet)
import librosa
import numpy as np
import pyworld
from scipy.interpolate import interp1d
import torch


class LogMelFBank():

    def __init__(
        self,
        sr,
        n_fft,
        hop_length,
        win_length,
        n_mels,
        fmin=0,
        fmax=None,
    ):
        """Melspectrogram extractor.

        Args:
            sr (int): sampling rate of the incoming signal
            n_fft (int): number of FFT components
            hop_length (int):  the distance between neighboring sliding window frames
            win_length (int): the size of window frame and STFT filter
            n_mels (int): number of Mel bands to generate
            fmin (int): lowest frequency (in Hz)
            fmax (int): highest frequency (in Hz)
        """
        super().__init__()
        self.sr = sr
        # stft
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length

        # mel
        self.n_mels = n_mels

        self.window = torch.hann_window(win_length)
        self.mel_filter = torch.from_numpy(
            librosa.filters.mel(sr=sr,
                                n_fft=n_fft,
                                n_mels=n_mels,
                                fmin=fmin,
                                fmax=fmax))

    def get_mel_spectrogram(self, wav):
        return torch_melspectrogram(wav, self.n_fft, self.hop_length,
                                    self.win_length, self.window,
                                    self.mel_filter)

    def get_linear_spectrogram(self, wav):
        return torch_linear_spectrogram(wav, self.n_fft, self.hop_length,
                                        self.win_length, self.window)


class Pitch():

    def __init__(self, sr, hop_length, pitch_min, pitch_max):

        self.sr = sr
        self.hop_length = hop_length
        self.pitch_min = pitch_min
        self.pitch_max = pitch_max

    def _convert_to_continuous_pitch(self, pitch: np.array) -> np.array:
        if (pitch == 0).all():
            print("All frames seems to be unvoiced.")
            return pitch

        # padding start and end of pitch sequence
        start_pitch = pitch[pitch != 0][0]
        end_pitch = pitch[pitch != 0][-1]
        start_idx = np.where(pitch == start_pitch)[0][0]
        end_idx = np.where(pitch == end_pitch)[0][-1]
        pitch[:start_idx] = start_pitch
        pitch[end_idx:] = end_pitch

        # get non-zero frame index
        nonzero_idxs = np.where(pitch != 0)[0]

        # perform linear interpolation
        interp_fn = interp1d(nonzero_idxs, pitch[nonzero_idxs])
        pitch = interp_fn(np.arange(0, pitch.shape[0]))

        return pitch

    def _calculate_pitch(self,
                         input: np.array,
                         use_continuous_pitch=True,
                         use_log_pitch=False) -> np.array:
        input = input.astype(np.float)
        frame_period = 1000 * self.hop_length / self.sr

        pitch, timeaxis = pyworld.dio(input,
                                      fs=self.sr,
                                      f0_floor=self.pitch_min,
                                      f0_ceil=self.pitch_max,
                                      frame_period=frame_period)
        pitch = pyworld.stonemask(input, pitch, timeaxis, self.sr)
        if use_continuous_pitch:
            pitch = self._convert_to_continuous_pitch(pitch)
        if use_log_pitch:
            nonzero_idxs = np.where(pitch != 0)[0]
            pitch[nonzero_idxs] = np.log(pitch[nonzero_idxs])
        return pitch.reshape(-1)

    def _average_by_duration(self, input: np.array, d: np.array) -> np.array:
        d_cumsum = np.pad(d.cumsum(0), (1, 0), 'constant')
        arr_list = []
        for start, end in zip(d_cumsum[:-1], d_cumsum[1:]):
            arr = input[start:end]
            mask = arr == 0
            arr[mask] = 0
            avg_arr = np.mean(arr, axis=0) if len(arr) != 0 else np.array(0)
            arr_list.append(avg_arr)

        # shape : (T)
        arr_list = np.array(arr_list)

        return arr_list

    def get_pitch(self,
                  wav,
                  use_continuous_pitch=True,
                  use_log_pitch=False,
                  use_token_averaged_pitch=True,
                  duration=None):
        pitch = self._calculate_pitch(wav, use_continuous_pitch, use_log_pitch)
        if use_token_averaged_pitch and duration is not None:
            pitch = self._average_by_duration(pitch, duration)
        return pitch


class Energy():

    def __init__(self, sr, n_fft, hop_length, win_length, min_amp=1e-5):

        self.sr = sr
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.window = torch.hann_window(win_length)
        self.min_amp = min_amp

    def _stft(self, wav):
        D = librosa.core.stft(wav,
                              n_fft=self.n_fft,
                              hop_length=self.hop_length,
                              win_length=self.win_length,
                              window=self.window,
                              center=self.center,
                              pad_mode=self.pad_mode)
        return D

    def _calculate_energy(self, x):
        stft = torch_stft(x, self.n_fft, self.hop_length, self.win_length,
                          self.win_length)
        energy = (stft.abs()**2).sum(dim=0).clamp(min=self.min_amp).sqrt()
        return energy

    def _average_by_duration(self, input: np.array, d: np.array) -> np.array:
        d_cumsum = np.pad(d.cumsum(0), (1, 0), 'constant')
        arr_list = []
        for start, end in zip(d_cumsum[:-1], d_cumsum[1:]):
            arr = input[start:end]
            avg_arr = np.mean(arr, axis=0) if len(arr) != 0 else np.array(0)
            arr_list.append(avg_arr)
        # shape (T)
        arr_list = np.array(arr_list)
        return arr_list

    def get_energy(self, wav, use_token_averaged_energy=True, duration=None):
        energy = self._calculate_energy(wav)
        if use_token_averaged_energy and duration is not None:
            energy = self._average_by_duration(energy, duration)
        return energy


def torch_stft(x, n_fft, hop_length, win_length, window):
    """Performing STFT using torch.

    Args:
        x (torch.Tensor): input signal. Shape (*,t). * is an optional batch
        dimension.
        n_fft (int): size of Fourier transform.
        hop_length (int): the distance between neighboring sliding window
        frames.
        win_length (int): the size of window frame and STFT filter.
        window (torch.Tensor): window tensor. Shape (win_length).

    Returns: STFT of shape (*,1+n_fft/2,t)
    """
    return torch.stft(x,
                      n_fft=n_fft,
                      hop_length=hop_length,
                      win_length=win_length,
                      window=window,
                      center=True,
                      onesided=True,
                      return_complex=True)


def torch_melspectrogram(x,
                         n_fft,
                         hop_length,
                         win_length,
                         window,
                         mel_basis,
                         min_amp=1e-5):
    """Calculating melspectrogram using torch.

    Args:
        x (torch.Tensor): input signal. Shape (*,t). * is an optional batch
        dimension.
        n_fft (int): size of Fourier transform.
        hop_length (int): the distance between neighboring sliding window
        frames.
        win_length (int): the size of window frame and STFT filter.
        window (torch.Tensor): window tensor. Shape (win_length).
        mel_basis (torch.Tensor): mel filter-bank of shape (n_mels,1+n_fft/2).
        min_amp (float): minimum amplitude. Defaults to 1e-5.

    Returns: melspectrogram of shape (*,n_mels,t)
    """
    stft = torch_stft(x, n_fft, hop_length, win_length, window)
    spec = torch.matmul(mel_basis, torch.abs(stft))
    return torch.log10(torch.clamp(torch.abs(spec), min=min_amp))


def torch_linear_spectrogram(x,
                             n_fft,
                             hop_length,
                             win_length,
                             window,
                             min_amp=1e-5):
    """Calculating linear spectrogram using torch.

    Args:
        x (torch.Tensor): input signal. Shape (*,t). * is an optional batch
        dimension.
        n_fft (int): size of Fourier transform.
        hop_length (int): the distance between neighboring sliding window
        frames.
        win_length (int): the size of window frame and STFT filter.
        window (torch.Tensor): window tensor. Shape (win_length).
        min_amp (float): minimum amplitude. Defaults to 1e-5.

    Returns: spectrogram of shape (*,1+n_fft/2,t)
    """
    stft = torch_stft(x, n_fft, hop_length, win_length, window)
    return torch.log10(torch.clamp(torch.abs(stft), min=min_amp))
