#! /bin/bash

# some parameters
model="cavp"
id="_01"
batch_size=100

# set save paths
ckpt_path="log/log_"$model$id
if [ ! -d $ckpt_path ]; then
  mkdir -p $ckpt_path
fi

# save and continue training
if [ ! -f $ckpt_path"/infos.json" ]; then
  start_from=""
else
  start_from="--start_from "$ckpt_path
fi

# first XE train
python tools/train.py --id $model$id --caption_model $model --batch_size $batch_size --save_checkpoint_every -1 --rnn_size 1300 --input_encoding_size 1000 --att_hid_size 1024 --use_bn 1 --max_epochs 37 --scheduled_sampling_start 0 --learning_rate 5e-4 --learning_rate_decay_start 0 --language_eval 1 --checkpoint_path $ckpt_path $start_from

# second RL training
python tools/train.py --id $model$id --caption_model $model --batch_size $batch_size --save_checkpoint_every -1 --rnn_size 1300 --input_encoding_size 1000 --att_hid_size 1024 --use_bn 1 --self_critical_after 37 --scheduled_sampling_start 0 --learning_rate 5e-5 --learning_rate_decay_start 0 --learning_rate_decay_every 55 --learning_rate_decay_rate 0.1 --language_eval 1 --checkpoint_path $ckpt_path --start_from $ckpt_path
