#!/bin/bash
#SBATCH --partition=gpu_8
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --export=NONE
# BATCH=$1
# LR=$2
# TEMP=$3
# SIZE=$4
# PRODUCT=$5
# AUG=$6
BATCH=128
LR=0.00005
TEMP=0.07
SIZE="xlarge"
# PRODUCT=$5
AUG="all"
POSTAUG="all"
products=("watches" "shoes" "cameras")
# computers
for PRODUCT in "${products[@]}"
do
    CUDA_VISIBLE_DEVICES=3 python run_pretraining.py \
        --do_train \
        --train_file /scratch/mhussein/contrastive-product-matching/data/processed/wdc-lspc/contrastive/pre-train/"$PRODUCT"/"$PRODUCT"_train_$SIZE.pkl.gz \
        --id_deduction_set /scratch/mhussein/contrastive-product-matching/data/raw/wdc-lspc/training-sets/"$PRODUCT"_train_$SIZE.json.gz \
        --tokenizer="roberta-base" \
        --grad_checkpoint=True \
        --output_dir /scratch/mhussein/contrastive-product-matching/reports/contrastive/"$PRODUCT"-$SIZE-$AUG$BATCH-$LR-$TEMP-roberta-base/ \
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
    
    CUDA_VISIBLE_DEVICES=3 python run_finetune_siamese.py \
	--model_pretrained_checkpoint /scratch/mhussein/contrastive-product-matching/reports/contrastive/"$PRODUCT"-$SIZE-$AUG$BATCH-$LR-$TEMP-roberta-base/pytorch_model.bin \
    --do_train \
    --train_file /scratch/mhussein/contrastive-product-matching/data/interim/wdc-lspc/training-sets/preprocessed_"$PRODUCT"_train_$SIZE.pkl.gz \
	--train_size=$SIZE \
	--validation_file /scratch/mhussein/contrastive-product-matching/data/interim/wdc-lspc/training-sets/preprocessed_"$PRODUCT"_train_$SIZE.pkl.gz \
	--test_file /scratch/mhussein/contrastive-product-matching/data/interim/wdc-lspc/gold-standards/preprocessed_"$PRODUCT"_gs.pkl.gz \
	--evaluation_strategy=epoch \
	--tokenizer="roberta-base" \
	--grad_checkpoint=True \
    --output_dir /scratch/mhussein/contrastive-product-matching/reports/contrastive-ft-siamese/"$PRODUCT"-$SIZE-$POSTAUG$BATCH-$AUG$LR-$TEMP-frozen-roberta-base/ \
	--per_device_train_batch_size=64 \
	--learning_rate=5e-05 \
	--weight_decay=0.01 \
	--num_train_epochs=50 \
	--lr_scheduler_type="linear" \
	--warmup_ratio=0.05 \
	--max_grad_norm=1.0 \
	--fp16 \
	--metric_for_best_model=loss \
	--dataloader_num_workers=4 \
	--disable_tqdm=True \
	--save_strategy="epoch" \
	--load_best_model_at_end \
	--augment=$POSTAUG \

done

