#!/bin/bash
#SBATCH --partition=gpu_8
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --export=NONE
BATCH=$1
LR=$2
TEMP=$3
AUG=$4
CUDA_VISIBLE_DEVICES=1 python run_pretraining_deepmatcher_nosplit.py \
    --do_train \
	--dataset_name=musicbrainz20k \
	--clean=True \
    --train_file /scratch/mhussein/contrastive-product-matching/data/processed/musicbrainz20k/contrastive/musicbrainz20k-train.pkl.gz \
	--id_deduction_set /scratch/mhussein/contrastive-product-matching/data/interim/musicbrainz20k/musicbrainz20k-train.json.gz \
	--tokenizer="roberta-base" \
	--grad_checkpoint=True \
    --output_dir /scratch/mhussein/contrastive-product-matching/reports/contrastive/musicbrainz20k-$AUG$BATCH-$LR-$TEMP-roberta-base/ \
	--temperature=$TEMP \
	--per_device_train_batch_size=$BATCH \
	--learning_rate=$LR \
	--weight_decay=0.01 \
	--num_train_epochs=200 \
	--lr_scheduler_type="linear" \
	--warmup_ratio=0.05 \
	--max_grad_norm=1.0 \
	--fp16 \
	--dataloader_num_workers=4 \
	--disable_tqdm=True \
	--save_strategy="epoch" \
	--logging_strategy="epoch" \
	--augment=$AUG \
