# Copyright (c) 2022 Binbin Zhang(binbzha@qq.com), Jie Chen
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

import argparse
import pathlib
import collections
import shutil

import numpy as np
import torch
from yacs import config
import jsonlines
import librosa

from wetts.models.am.fastspeech2_emo.fastspeech2 import FastSpeech2
from wetts.bin.fastspeech2_train import load_ckpt
from wetts.utils.file_utils import read_key2id, read_lists
from wetts.dataset.feats import LogMelFBank
from wetts.utils import plot


def get_args(argv=None):
    parser = argparse.ArgumentParser(
        description='FastSpeech2 inference script.')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--datalist', required=True, help='test data list')
    parser.add_argument('--cmvn_dir',
                        required=True,
                        help='mel/energy/pitch cmvn dir')
    parser.add_argument(
        '--spk2id_file',
        type=str,
        required=True,
        help='path to spk2id file, this file must be provided '
        'for both multi-speaker FastSpeech2 and single-speaker '
        'FastSpeech2')
    parser.add_argument('--phn2id_file',
                        required=True,
                        help='phone to id file')
    parser.add_argument('--emo2id_file',
                        required=True,
                        help='emotion to id file')
    parser.add_argument('--special_token_file',
                        required=True,
                        help='special tokens file')
    parser.add_argument('--pitch_control',
                        type=float,
                        default=1.0,
                        help='Pitch manipulation factor.')
    parser.add_argument('--energy_control',
                        type=float,
                        default=1.0,
                        help='Energy manipulation factor.')
    parser.add_argument('--duration_control',
                        type=float,
                        default=1.0,
                        help='Duration manipulation factor.')
    parser.add_argument('--emotion_control',
                        type=float,
                        default=1.0,
                        help='Emotion manipulation factor')
    parser.add_argument('--export_dir', type=str, help='path to save mel spec')
    parser.add_argument('--ckpt', type=str, help='path to mel spectrogram')
    args = parser.parse_args(argv)
    return args


def load_ckpt(path):
    with open(path, 'rb') as fin:
        (model_state_dict, lr_scheduler_state_dict, optimizer_state_dict,
         train_step, val_step, epoch) = torch.load(fin, 'cpu')
        return (model_state_dict, lr_scheduler_state_dict,
                optimizer_state_dict, train_step, val_step, epoch)


def load_datalist(path, phn2id, spk2id, emo2id, special_tokens, sr, n_fft,
                  hop_length, win_length, n_mels):
    mel_extractor = LogMelFBank(sr, n_fft, hop_length, win_length, n_mels)
    datadict = collections.defaultdict(dict)
    with jsonlines.open(path) as f:
        for sample in f:
            if datadict[sample['speaker']].get(sample['emotion']) is None:
                datadict[sample['speaker']][sample['emotion']] = []
            datadict[sample['speaker']][sample['emotion']].append(sample)
    samples = {}
    for speaker in sorted(datadict):
        if speaker not in samples:
            samples[speaker] = {}
        for emotion in datadict[speaker]:
            # only take 1 sample from each emotion
            samples[speaker][emotion] = datadict[speaker][emotion][0]
            continue
    for speaker in samples:
        targets = []
        source = None
        for emo in samples[speaker]:
            if emo == 'Neutral':
                source = samples[speaker][emo]
            else:
                targets.append(samples[speaker][emo])
        for tgt in targets:
            source_phn_seq = source['text']
            source_original_text = source['original_text']
            source_emotion = source['emotion']
            source_language = source['language']
            source_id = source['key']
            source_wav_path = source['wav_path']
            source_phn_id_seq = torch.tensor(
                [[phn2id[x] for x in source_phn_seq]], dtype=torch.long)
            source_speaker = source['speaker']
            source_speaker_id = torch.tensor([spk2id[source_speaker]],
                                             dtype=torch.long)
            source_phn_seq_length = torch.tensor([len(source_phn_seq)],
                                                 dtype=torch.long)
            source_token_type = torch.tensor(
                [[0 if x in special_tokens else 1 for x in source_phn_seq]],
                dtype=torch.long)

            target_phn_seq = tgt['text']
            target_original_text = tgt['original_text']
            target_emotion = tgt['emotion']
            target_language = tgt['language']
            target_id = tgt['key']
            target_wav_path = tgt['wav_path']
            target_phn_id_seq = torch.tensor(
                [[phn2id[x] for x in target_phn_seq]], dtype=torch.long)
            target_speaker = source['speaker']
            target_speaker_id = torch.tensor([spk2id[target_speaker]],
                                             dtype=torch.long)
            target_phn_seq_length = torch.tensor([len(target_phn_seq)],
                                                 dtype=torch.long)
            tgt_wav,_ = librosa.load(target_wav_path, sr=sr)
            tgt_mel = mel_extractor.get_mel_spectrogram(torch.from_numpy(tgt_wav)).T.unsqueeze(0)
            tgt_mel_len = torch.tensor([tgt_mel.size(1)], dtype=torch.long)
            yield {
                'src_phn_seq': source_phn_seq,
                'src_ori_txt': source_original_text,
                'src_emo': source_emotion,
                'src_lan': source_language,
                'src_id': source_id,
                'src_phn_id': source_phn_id_seq,
                'src_wav_path': source_wav_path,
                'src_spk_id': source_speaker_id,
                'src_phn_seq_len': source_phn_seq_length,
                'src_spk': source_speaker,
                'src_token_type': source_token_type,
                'tgt_phn_seq': target_phn_seq,
                'tgt_ori_txt': target_original_text,
                'tgt_emo': target_emotion,
                'tgt_lan': target_language,
                'tgt_id': target_id,
                'tgt_wav_path': target_wav_path,
                'tgt_phn_seq': target_phn_id_seq,
                'tgt_spk': target_speaker_id,
                'tgt_phn_seq_len': target_phn_seq_length,
                'tgt_mel_len': tgt_mel_len,
                'tgt_mel': tgt_mel
            }


def main(args):
    with open(args.config, 'r') as fin:
        conf = config.load_cfg(fin)
    export_dir = pathlib.Path(args.export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    cmvn_dir = pathlib.Path(args.cmvn_dir)

    mel_stats = np.loadtxt(cmvn_dir / 'mel_cmvn.txt')
    pitch_stats = np.loadtxt(cmvn_dir / 'pitch_cmvn.txt')
    energy_stats = np.loadtxt(cmvn_dir / 'energy_cmvn.txt')

    mel_mean, mel_sigma = mel_stats
    pitch_mean, pitch_sigma, pitch_min, pitch_max = pitch_stats
    energy_mean, energy_sigma, energy_min, energy_max = energy_stats

    phn2id = read_key2id(args.phn2id_file)
    spk2id = read_key2id(args.spk2id_file)
    emo2id = read_key2id(args.emo2id_file)
    special_token = read_lists(args.special_token_file)

    model = FastSpeech2(
        conf.model.d_model, conf.model.n_enc_layer, conf.model.n_enc_head,
        conf.model.n_enc_conv_filter, conf.model.enc_conv_kernel_size,
        conf.model.enc_dropout, len(phn2id), conf.model.padding_idx,
        conf.model.n_va_conv_filter, conf.model.va_conv_kernel_size,
        conf.model.va_dropout, pitch_min, pitch_max, pitch_mean, pitch_sigma,
        energy_min, energy_max, energy_mean, energy_sigma,
        conf.model.n_pitch_bin, conf.model.n_energy_bin,
        conf.model.n_dec_layer, conf.model.n_dec_head,
        conf.model.n_dec_conv_filter,
        conf.model.dec_conv_kernel_size, conf.model.dec_dropout, conf.n_mels,
        len(spk2id), conf.model.postnet_kernel_size,
        conf.model.postnet_hidden_dim, conf.model.n_postnet_conv_layers,
        conf.model.postnet_dropout, conf.model.max_pos_enc_len)

    model_state_dict, _, _, _, _, epoch = load_ckpt(args.ckpt)
    print(f'loading FastSpeech2 from epoch {epoch}')
    model.load_state_dict(model_state_dict)

    with torch.no_grad():
        with jsonlines.open(export_dir / 'fastspeech2_mel_prediction.jsonl',
                            'w') as f:
            for sample in load_datalist(args.datalist, phn2id, spk2id, emo2id,
                                        special_token, conf.sr, conf.n_fft,
                                        conf.hop_length, conf.win_length,
                                        conf.n_mels):
                phn_seq = sample['src_phn_id']
                spk_id = sample['src_spk_id']
                phn_len = sample['src_phn_seq_len']
                token_type = sample['src_token_type']
                ref_mel = sample['tgt_mel']
                ref_mel_len = sample['tgt_mel_len']
                _, postnet_mel_prediction, *_ = model(phn_seq, phn_len,
                                                      token_type, ref_mel,
                                                      ref_mel_len, speaker=spk_id)
                title = (f'{sample["src_id"]}'
                         f'__{sample["tgt_id"]}_{sample["tgt_lan"]}_{sample["tgt_emo"]}')
                export_name = export_dir / f'{title}.npy'
                print(f'saving {export_name}')
                postnet_mel_prediction = postnet_mel_prediction[0].cpu().numpy(
                )
                np.save(export_name, postnet_mel_prediction)
                fig = plot.plot_mel([postnet_mel_prediction], [title])
                fig.savefig(export_dir / f'{title}.png')
                plot.plt.close()
                f.write({
                    'mel_prediction_filepath': str(export_name),
                    'ori_text': sample['src_ori_txt']
                })
                shutil.copy(sample['src_wav_path'],
                            export_dir / f'{sample["src_id"]}_{sample["src_emo"]}.wav')
                shutil.copy(sample['tgt_wav_path'],
                            export_dir / f'{sample["tgt_id"]}_{sample["tgt_emo"]}.wav')


if __name__ == '__main__':
    main(get_args())
