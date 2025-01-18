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
PREAUG=$5
python run_finetune_siamese.py \
	--model_pretrained_checkpoint /scratch/mhussein/contrastive-product-matching/reports/contrastive/dblpscholar-ssv-$PREAUG$BATCH-$LR-$TEMP-roberta-base/pytorch_model.bin \
    --do_train \
	--dataset_name=dblp-scholar \
	--frozen=False \
    --train_file /scratch/mhussein/contrastive-product-matching/data/interim/dblp-scholar/dblp-scholar-train.json.gz \
	--validation_file /scratch/mhussein/contrastive-product-matching/data/interim/dblp-scholar/dblp-scholar-train.json.gz \
	--test_file /scratch/mhussein/contrastive-product-matching/data/interim/dblp-scholar/dblp-scholar-gs.json.gz \
	--evaluation_strategy=epoch \
	--tokenizer="roberta-base" \
	--grad_checkpoint=True \
    --output_dir /scratch/mhussein/contrastive-product-matching/reports/contrastive-ft-siamese/dblpscholar-ssv-$AUG$BATCH-$PREAUG$LR-$TEMP-unfrozen-roberta-base/ \
	--per_device_train_batch_size=64 \
	--learning_rate=5e-05 \
	--weight_decay=0.01 \
	--num_train_epochs=50 \
	--lr_scheduler_type="linear" \
	--warmup_ratio=0.05 \
	--max_grad_norm=1.0 \
	--fp16 \
	--metric_for_best_model=f1 \
	--dataloader_num_workers=4 \
	--disable_tqdm=True \
	--save_strategy="epoch" \
	--load_best_model_at_end \
	--augment=$AUG \
	#--do_param_opt \