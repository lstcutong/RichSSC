#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1 python ./train.py \
--model='aic' \
--dataset=nyucad \
--batch_size=8 \
--workers=8 \
--lr=0.001 \
--use_cls=False \
--pretrain=./model_zoo/AICNet.pth.tar \
--checkpoint=./pretrain_model/res_aicnet \
--fc_nblocks=3 \
--model_name='ReS_SSC' 2>&1 |tee train_ReS_SSC_NYUCAD.log



