#!/usr/bin/env bash
# Copyright 2022 Binbin Zhang(binbzha@qq.com), Jie Chen
. path.sh

STAGE=                            # start from -1 if you need to download data
STOP_STAGE=

DATASET_URL=https://openslr.magicdatatech.com/resources/93/data_aishell3.tgz
DATASET_DIR=./dataset        # path to dataset directory

FASTSPEECH2_DIR=fastspeech2
FASTSPEECH2_FEATURE_DIR=$FASTSPEECH2_DIR/features
FASTSPEECH2_CONFIG=conf/fastspeech2.yaml

HIFIGAN_DIR=hifigan
HIFIGAN_CONFIG=conf/hifigan_v1.yaml


conda_base=$(conda info --base)
source $conda_base/bin/activate wetts

mkdir -p $FASTSPEECH2_FEATURE_DIR

if [ ${STAGE} -le -1 ] && [ ${STOP_STAGE} -ge -1 ]; then
  # Download data
  local/download_data.sh $DATASET_URL $DATASET_DIR
fi


if [ ${STAGE} -le 0 ] && [ ${STOP_STAGE} -ge 0 ]; then
  # Prepare wav.txt, speaker.txt and text.txt
  python local/prepare_data_list.py $DATASET_DIR $FASTSPEECH2_FEATURE_DIR/wav.txt \
    $FASTSPEECH2_FEATURE_DIR/speaker.txt $FASTSPEECH2_FEATURE_DIR/text.txt
  # Prepare special_tokens. Special tokens in AISHELL3 are % $.
  (echo %; echo $; echo sp; echo sil;) > $FASTSPEECH2_FEATURE_DIR/special_token.txt
  # Prepare lexicon.
  python tools/gen_mfa_pinyin_lexicon.py --with-tone --with-r \
    $FASTSPEECH2_FEATURE_DIR/lexicon.txt $FASTSPEECH2_FEATURE_DIR/phone.txt
  # Convert text in text.txt to phonemes.
  python local/convert_text_to_phn.py $FASTSPEECH2_FEATURE_DIR/text.txt \
    $FASTSPEECH2_FEATURE_DIR/lexicon.txt $FASTSPEECH2_FEATURE_DIR/special_token.txt \
    $FASTSPEECH2_FEATURE_DIR/text.txt
fi


if [ ${STAGE} -le 1 ] && [ ${STOP_STAGE} -ge 1 ]; then
  # Prepare alignment lab and pronounciation dictionary for MFA tools
  python local/prepare_alignment.py $FASTSPEECH2_FEATURE_DIR/wav.txt \
    $FASTSPEECH2_FEATURE_DIR/speaker.txt $FASTSPEECH2_FEATURE_DIR/text.txt \
    $FASTSPEECH2_FEATURE_DIR/special_token.txt \
    $FASTSPEECH2_FEATURE_DIR/mfa_pronounciation_dict.txt \
    $FASTSPEECH2_FEATURE_DIR/lab/
fi


if [ ${STAGE} -le 2 ] && [ ${STOP_STAGE} -ge 2 ]; then
  # MFA alignment
  mfa train -j 16 --phone_set PINYIN --overwrite \
      -a $DATASET_DIR/train/wav -t $FASTSPEECH2_FEATURE_DIR/mfa_temp \
      $FASTSPEECH2_FEATURE_DIR/lab \
      $FASTSPEECH2_FEATURE_DIR/mfa_pronounciation_dict.txt \
      -o $FASTSPEECH2_FEATURE_DIR/mfa/mfa_model.zip $FASTSPEECH2_FEATURE_DIR/TextGrid \
      --clean
fi


if [ ${STAGE} -le 3 ] && [ ${STOP_STAGE} -ge 3 ]; then
  python tools/gen_alignment_from_textgrid.py \
    $FASTSPEECH2_FEATURE_DIR/wav.txt \
    $FASTSPEECH2_FEATURE_DIR/speaker.txt $FASTSPEECH2_FEATURE_DIR/text.txt \
    $FASTSPEECH2_FEATURE_DIR/special_token.txt $FASTSPEECH2_FEATURE_DIR/TextGrid \
    $FASTSPEECH2_FEATURE_DIR/aligned_wav.txt \
    $FASTSPEECH2_FEATURE_DIR/aligned_speaker.txt \
    $FASTSPEECH2_FEATURE_DIR/duration.txt \
    $FASTSPEECH2_FEATURE_DIR/aligned_text.txt
  # speaker to id map
  cat $FASTSPEECH2_FEATURE_DIR/aligned_speaker.txt | awk '{print $1}' | sort | uniq | \
      awk '{print $1, NR-1}' > $FASTSPEECH2_FEATURE_DIR/spk2id
  # phone to id map
  python tools/gen_phn2id.py $FASTSPEECH2_FEATURE_DIR/lexicon.txt \
    $FASTSPEECH2_FEATURE_DIR/special_token.txt $FASTSPEECH2_FEATURE_DIR/phn2id
fi


if [ ${STAGE} -le 4 ] && [ ${STOP_STAGE} -ge 4 ]; then
  # generate training, validation and test samples
  python local/train_val_test_split.py $FASTSPEECH2_FEATURE_DIR/aligned_wav.txt \
  $FASTSPEECH2_FEATURE_DIR/aligned_speaker.txt \
  $FASTSPEECH2_FEATURE_DIR/aligned_text.txt \
  $FASTSPEECH2_FEATURE_DIR/duration.txt $FASTSPEECH2_FEATURE_DIR
fi


if [ ${STAGE} -le 5 ] && [ ${STOP_STAGE} -ge 5 ]; then
  # Prepare training samples
  python local/make_data_list.py $FASTSPEECH2_FEATURE_DIR/train/train_wav.txt \
      $FASTSPEECH2_FEATURE_DIR/train/train_speaker.txt \
      $FASTSPEECH2_FEATURE_DIR/train/train_text.txt \
      $FASTSPEECH2_FEATURE_DIR/train/train_duration.txt \
      $FASTSPEECH2_FEATURE_DIR/train/datalist.jsonl
  # Prepare validation samples
  python local/make_data_list.py $FASTSPEECH2_FEATURE_DIR/val/val_wav.txt \
      $FASTSPEECH2_FEATURE_DIR/val/val_speaker.txt \
      $FASTSPEECH2_FEATURE_DIR/val/val_text.txt \
      $FASTSPEECH2_FEATURE_DIR/val/val_duration.txt \
      $FASTSPEECH2_FEATURE_DIR/val/datalist.jsonl
  # Prepare test samples
  python local/make_data_list.py $FASTSPEECH2_FEATURE_DIR/test/test_wav.txt \
      $FASTSPEECH2_FEATURE_DIR/test/test_speaker.txt \
      $FASTSPEECH2_FEATURE_DIR/test/test_text.txt \
      $FASTSPEECH2_FEATURE_DIR/test/test_duration.txt \
      $FASTSPEECH2_FEATURE_DIR/test/datalist.jsonl
fi


if [ ${STAGE} -le 6 ] && [ ${STOP_STAGE} -ge 6 ]; then
  # Compute mel, f0, energy CMVN
  total=$(cat $FASTSPEECH2_FEATURE_DIR/train/datalist.jsonl | wc -l)
  python wetts/bin/compute_cmvn.py --total $total --num_workers 32 \
      $FASTSPEECH2_CONFIG $FASTSPEECH2_FEATURE_DIR/train/datalist.jsonl \
      $FASTSPEECH2_FEATURE_DIR/train
fi


if [ ${STAGE} -le 7 ] && [ ${STOP_STAGE} -ge 7 ]; then
  # train fastspeech2
  EPOCH=200
  python wetts/bin/train.py fastspeech2 --num_workers 32 \
      --config $FASTSPEECH2_CONFIG \
      --train_data_list $FASTSPEECH2_FEATURE_DIR/train/datalist.jsonl \
      --val_data_list $FASTSPEECH2_FEATURE_DIR/val/datalist.jsonl \
      --cmvn_dir $FASTSPEECH2_FEATURE_DIR/train \
      --spk2id_file $FASTSPEECH2_FEATURE_DIR/spk2id \
      --phn2id_file $FASTSPEECH2_FEATURE_DIR/phn2id \
      --special_tokens_file $FASTSPEECH2_FEATURE_DIR/special_token.txt \
      --log_dir $FASTSPEECH2_DIR/log/ \
      --batch_size 128 \
      --epoch $EPOCH
fi

FASTSPEECH2_INFERENCE_OUTPUTDIR=$FASTSPEECH2_DIR/inference_mels # path to directory for inferenced mels
if [ ${STAGE} -le 8 ] && [ ${STOP_STAGE} -ge 8 ]; then
  # inference fastspeech2
  TEXT_FILE=test_samples.txt                 # path to text file, each line contains one sample for inference
  SPEAKER_FILE=test_samples_speakers.txt     # path to speaker file, each line contains one speaker name for corresponding line in text file
  FASTSPEECH2_CKPT_PATH=                     # path to fastspeech2 checkpoint file
  python wetts/bin/inference.py fastspeech2 \
      --num_workers 4 \
      --batch_size 64 \
      --config $FASTSPEECH2_CONFIG \
      --text_file $TEXT_FILE \
      --speaker_file $SPEAKER_FILE \
      --lexicon_file $FASTSPEECH2_FEATURE_DIR/lexicon.txt \
      --cmvn_dir $FASTSPEECH2_FEATURE_DIR/train \
      --spk2id_file $FASTSPEECH2_FEATURE_DIR/spk2id \
      --phn2id_file $FASTSPEECH2_FEATURE_DIR/phn2id \
      --special_token_file $FASTSPEECH2_FEATURE_DIR/special_token.txt \
      --export_dir $FASTSPEECH2_INFERENCE_OUTPUTDIR \
      --ckpt $FASTSPEECH2_CKPT_PATH
fi


if [ ${STAGE} -le 9 ] && [ ${STOP_STAGE} -ge 9 ]; then
  # train hifigan using fastspeech2 training dataset
  EPOCH=100                              # Number of epoch for training hifigan
  python wetts/bin/hifigan_train.py train \
      --num_workers 32 \
      --batch_size_hifigan 64 \
      --fastspeech2_train_datalist $FASTSPEECH2_FEATURE_DIR/train/datalist.jsonl \
      --fastspeech2_val_datalist $FASTSPEECH2_FEATURE_DIR/val/datalist.jsonl \
      --hifigan_config $HIFIGAN_CONFIG \
      --epoch $EPOCH \
      --export_dir $HIFIGAN_DIR/train/
fi

if [ ${STAGE} -le 10 ] && [ ${STOP_STAGE} -ge 10 ]; then
  # finetune hifigan using fastspeech2 training dataset
  FASTSPEECH2_CKPT_PATH=          # path to fastspeech2 checkpoint
  HIFIGAN_CKPT_PATH=              # path to hifigan generator and discriminator checkpoint
                                  # e.g. $HIFIGAN_CKPT_PATH='g_02500000 do_02500000'
                                  # pretrained hifigan checkpoint can be obtained from:
                                  # https://github.com/jik876/hifi-gan
  # This stage will continue training hifigan using checkpoints specified with --hifigan_ckpt.
  # Mels predicted by fastspeech2 and ground truth audios from fastspeech2 training dataset are
  # used as training dataset for hifigan.
  # Commenting out --hifigan_ckpt will train hifigan from scratch.
  # When running finetune command for the first time, --generate_samples should be specified.
  EPOCH=                          # number of epoch for finetune
  python wetts/bin/hifigan_train.py finetune \
      --num_workers 32 \
      --batch_size_hifigan 32 \
      --batch_size_fastspeech2 32 \
      --fastspeech2_config $FASTSPEECH2_CONFIG \
      --fastspeech2_train_datalist $FASTSPEECH2_FEATURE_DIR/train/datalist.jsonl \
      --HIFIGAN_CONFIG $HIFIGAN_CONFIG \
      --phn2id_file $FASTSPEECH2_FEATURE_DIR/phn2id \
      --spk2id_file $FASTSPEECH2_FEATURE_DIR/spk2id \
      --special_tokens_file $FASTSPEECH2_FEATURE_DIR/special_token.txt \
      --cmvn_dir $FASTSPEECH2_FEATURE_DIR/train \
      --fastspeech2_ckpt $FASTSPEECH2_CKPT_PATH \
      --hifigan_ckpt $HIFIGAN_CKPT_PATH \
      --epoch $EPOCH \
      --export_dir $HIFIGAN_DIR/finetune \
      --generate_samples
fi


if [ ${STAGE} -le 11 ] && [ ${STOP_STAGE} -ge 11 ]; then
  # hifigan inference
  HIFIGAN_GENERATOR_CKPT_PATH=    # path to hifigan generator checkpoint
                                  # e.g. $HIFIGAN_GENERATOR_CKPT_PATH=g_02500000
                                  # pretrained hifigan checkpoint can be obtained from:
                                  # https://github.com/jik876/hifi-gan

  CUDA_VISIBLE_DEVICES=4 python wetts/bin/hifigan_inference.py \
      --num_workers 4 \
      --batch_size 32 \
      --config $HIFIGAN_CONFIG \
      --datalist $FASTSPEECH2_INFERENCE_OUTPUTDIR/fastspeech2_mel_prediction.jsonl \
      --ckpt $HIFIGAN_GENERATOR_CKPT_PATH \
      --export_dir $HIFIGAN_DIR/predicted_wav
fi
